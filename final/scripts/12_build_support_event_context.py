#!/usr/bin/env python3
"""
12_build_support_event_context.py

Build a light event-context table for support roaming labels.

The output is one row per (match_id, team_id). It uses Riot timeline events as
target-construction evidence only. These columns must not be used as pregame
model features.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_SCRIPT_DIR = REPO_ROOT / "ProgresoActual2" / "scripts"
sys.path.insert(0, str(GEOMETRY_SCRIPT_DIR))

from build_geometry_v5_frame_state_distributions import classify_chunk_absolute  # noqa: E402


DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw" / "raw" / "europe"
DEFAULT_CONFIG = REPO_ROOT / "ProgresoActual2" / "data" / "geometry" / "manual_geometry_v5_config.json"
DEFAULT_OUTDIR = REPO_ROOT / "final" / "data" / "event_context"

JOIN_KEYS = ["match_id", "team_id"]
BOT_CONTEXT_ZONES = {"BOT_LANE_CORE", "BOT_SIDE_NEAR", "RIVER_BOT", "DRAGON_AREA"}
BASE_ZONES = {"BLUE_BASE", "RED_BASE"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build support event context from Riot timelines.")
    p.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--out-name", default="support_event_context_m12.parquet")
    p.add_argument("--max-minute", type=float, default=12.0)
    p.add_argument("--chunk-size", type=int, default=250000)
    p.add_argument("--limit-matches", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def match_dirs(raw_root: Path, limit: int) -> Iterable[Path]:
    dirs = sorted(p for p in raw_root.iterdir() if p.is_dir())
    if limit and limit > 0:
        dirs = dirs[:limit]
    return dirs


def participant_context(match: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for p in match.get("info", {}).get("participants", []):
        pid = int(p.get("participantId", 0) or 0)
        out[pid] = {
            "team_id": int(p.get("teamId", 0) or 0),
            "role": str(p.get("teamPosition", "") or ""),
            "champion_name": p.get("championName"),
        }
    return out


def team_roles(pctx: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    teams: Dict[int, Dict[str, Any]] = {
        100: {"team_id": 100, "side": "blue"},
        200: {"team_id": 200, "side": "red"},
    }
    for pid, info in pctx.items():
        team_id = info["team_id"]
        if team_id not in teams:
            continue
        role = info["role"]
        if role == "UTILITY":
            teams[team_id]["support_pid"] = pid
            teams[team_id]["support_champion_name"] = info["champion_name"]
        elif role == "BOTTOM":
            teams[team_id]["adc_pid"] = pid
            teams[team_id]["adc_champion_name"] = info["champion_name"]
    return teams


def event_minute(event: Dict[str, Any]) -> float:
    return float(event.get("timestamp", 0) or 0) / 60000.0


def position_tuple(event: Dict[str, Any]) -> Tuple[float, float] | None:
    pos = event.get("position")
    if not isinstance(pos, dict):
        return None
    x = pos.get("x")
    y = pos.get("y")
    if x is None or y is None:
        return None
    return float(x), float(y)


def zone_order(config: dict) -> List[str]:
    order = ["OUT_OF_MAP", "UNCLASSIFIED"] + list(config["colors"].keys())
    for zone in config["priority"]:
        if zone not in order:
            order.append(zone)
    return order


def classify_event_positions(pos_events: pd.DataFrame, config: dict, chunk_size: int) -> pd.DataFrame:
    if pos_events.empty:
        pos_events["zone_v6"] = []
        pos_events["out_bot_context_v6"] = []
        return pos_events

    order = zone_order(config)
    zone_to_id = {zone: idx for idx, zone in enumerate(order)}
    id_to_zone = np.asarray(order, dtype=object)
    x = pos_events["x"].to_numpy(dtype=np.float64)
    y = pos_events["y"].to_numpy(dtype=np.float64)
    zone_ids = np.empty(x.shape[0], dtype=np.int16)
    for start in range(0, len(x), chunk_size):
        end = min(start + chunk_size, len(x))
        zone_ids[start:end] = classify_chunk_absolute(x[start:end], y[start:end], config, zone_to_id)
        print(f"[Classify events] rows {end:,}/{len(x):,}")
    out = pos_events.copy()
    out["zone_v6"] = id_to_zone[zone_ids]
    out["out_bot_context_v6"] = ~out["zone_v6"].isin(BOT_CONTEXT_ZONES | BASE_ZONES)
    out["in_bot_context_v6"] = out["zone_v6"].isin(BOT_CONTEXT_ZONES)
    return out


def empty_row(match_id: str, team: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "match_id": match_id,
        "team_id": team["team_id"],
        "side": team["side"],
        "support_participant_id": team.get("support_pid"),
        "adc_participant_id": team.get("adc_pid"),
        "support_champion_name": team.get("support_champion_name"),
        "adc_champion_name": team.get("adc_champion_name"),
        "support_wards_0_12": 0,
        "support_ward_kills_0_12": 0,
        "support_kill_assists_0_12": 0,
        "support_deaths_0_12": 0,
        "adc_deaths_0_12": 0,
        "team_objectives_0_12": 0,
        "team_dragons_0_12": 0,
        "support_building_kills_0_12": 0,
        "support_plate_destroys_0_12": 0,
        "events_with_position_0_12": 0,
    }
    return row


def process_match(match_dir: Path, max_minute: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    match_path = match_dir / "match.json"
    timeline_path = match_dir / "timeline.json"
    if not match_path.exists() or not timeline_path.exists():
        return [], []

    try:
        match = load_json(match_path)
        timeline = load_json(timeline_path)
    except Exception:
        return [], []
    match_id = str(match.get("metadata", {}).get("matchId") or match_dir.name)
    pctx = participant_context(match)
    teams = team_roles(pctx)
    rows_by_team = {
        team_id: empty_row(match_id, team)
        for team_id, team in teams.items()
        if team.get("support_pid") and team.get("adc_pid")
    }
    if not rows_by_team:
        return [], []

    team_by_pid = {pid: info["team_id"] for pid, info in pctx.items()}
    pos_events: List[Dict[str, Any]] = []
    max_ts = max_minute * 60000.0

    for frame in timeline.get("info", {}).get("frames", []):
        for event in frame.get("events", []):
            ts = float(event.get("timestamp", 0) or 0)
            if ts < 0 or ts > max_ts:
                continue
            typ = event.get("type")
            minute = ts / 60000.0

            if typ == "WARD_PLACED":
                creator = int(event.get("creatorId", 0) or 0)
                for team_id, team in teams.items():
                    if creator == team.get("support_pid") and team_id in rows_by_team:
                        rows_by_team[team_id]["support_wards_0_12"] += 1

            elif typ == "WARD_KILL":
                killer = int(event.get("killerId", 0) or 0)
                for team_id, team in teams.items():
                    if killer == team.get("support_pid") and team_id in rows_by_team:
                        rows_by_team[team_id]["support_ward_kills_0_12"] += 1

            elif typ == "CHAMPION_KILL":
                killer = int(event.get("killerId", 0) or 0)
                victim = int(event.get("victimId", 0) or 0)
                assists = {int(x) for x in event.get("assistingParticipantIds", []) or []}
                pos = position_tuple(event)
                for team_id, team in teams.items():
                    if team_id not in rows_by_team:
                        continue
                    support_pid = team.get("support_pid")
                    adc_pid = team.get("adc_pid")
                    support_involved = support_pid == killer or support_pid in assists
                    if support_involved:
                        rows_by_team[team_id]["support_kill_assists_0_12"] += 1
                        if pos:
                            pos_events.append({
                                "match_id": match_id, "team_id": team_id,
                                "event_kind": "support_kill_assist", "minute": minute,
                                "x": pos[0], "y": pos[1],
                            })
                    if victim == support_pid:
                        rows_by_team[team_id]["support_deaths_0_12"] += 1
                        if pos:
                            pos_events.append({
                                "match_id": match_id, "team_id": team_id,
                                "event_kind": "support_death", "minute": minute,
                                "x": pos[0], "y": pos[1],
                            })
                    if victim == adc_pid:
                        rows_by_team[team_id]["adc_deaths_0_12"] += 1
                        if pos:
                            pos_events.append({
                                "match_id": match_id, "team_id": team_id,
                                "event_kind": "adc_death", "minute": minute,
                                "x": pos[0], "y": pos[1],
                            })

            elif typ == "ELITE_MONSTER_KILL":
                killer_team = int(event.get("killerTeamId") or team_by_pid.get(int(event.get("killerId", 0) or 0), 0) or 0)
                if killer_team in rows_by_team:
                    rows_by_team[killer_team]["team_objectives_0_12"] += 1
                    if event.get("monsterType") == "DRAGON":
                        rows_by_team[killer_team]["team_dragons_0_12"] += 1
                    pos = position_tuple(event)
                    if pos:
                        pos_events.append({
                            "match_id": match_id, "team_id": killer_team,
                            "event_kind": "team_objective", "minute": minute,
                            "x": pos[0], "y": pos[1],
                        })

            elif typ == "BUILDING_KILL":
                killer = int(event.get("killerId", 0) or 0)
                assists = {int(x) for x in event.get("assistingParticipantIds", []) or []}
                pos = position_tuple(event)
                for team_id, team in teams.items():
                    if team_id not in rows_by_team:
                        continue
                    support_pid = team.get("support_pid")
                    if support_pid == killer or support_pid in assists:
                        rows_by_team[team_id]["support_building_kills_0_12"] += 1
                        if pos:
                            pos_events.append({
                                "match_id": match_id, "team_id": team_id,
                                "event_kind": "support_building_kill", "minute": minute,
                                "x": pos[0], "y": pos[1],
                            })

            elif typ == "TURRET_PLATE_DESTROYED":
                killer = int(event.get("killerId", 0) or 0)
                pos = position_tuple(event)
                for team_id, team in teams.items():
                    if team_id not in rows_by_team:
                        continue
                    if killer == team.get("support_pid"):
                        rows_by_team[team_id]["support_plate_destroys_0_12"] += 1
                        if pos:
                            pos_events.append({
                                "match_id": match_id, "team_id": team_id,
                                "event_kind": "support_plate_destroy", "minute": minute,
                                "x": pos[0], "y": pos[1],
                            })

    return list(rows_by_team.values()), pos_events


def process_match_chunk(match_dirs_chunk: List[Path], max_minute: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    pos_rows: List[Dict[str, Any]] = []
    for match_dir in match_dirs_chunk:
        r, p = process_match(match_dir, max_minute)
        rows.extend(r)
        pos_rows.extend(p)
    return rows, pos_rows


def aggregate_position_events(pos_events: pd.DataFrame) -> pd.DataFrame:
    base_cols = JOIN_KEYS + [
        "support_kill_assists_out_bot_0_12",
        "support_kill_assists_bot_0_12",
        "support_deaths_out_bot_0_12",
        "support_deaths_bot_0_12",
        "adc_deaths_bot_0_12",
        "team_objectives_out_bot_0_12",
        "support_building_kills_out_bot_0_12",
        "support_building_kills_bot_0_12",
        "support_plate_destroys_out_bot_0_12",
        "support_plate_destroys_bot_0_12",
    ]
    if pos_events.empty:
        return pd.DataFrame(columns=base_cols)

    rows = []
    for (match_id, team_id), g in pos_events.groupby(JOIN_KEYS):
        rows.append({
            "match_id": match_id,
            "team_id": team_id,
            "support_kill_assists_out_bot_0_12": int(((g["event_kind"] == "support_kill_assist") & g["out_bot_context_v6"]).sum()),
            "support_kill_assists_bot_0_12": int(((g["event_kind"] == "support_kill_assist") & g["in_bot_context_v6"]).sum()),
            "support_deaths_out_bot_0_12": int(((g["event_kind"] == "support_death") & g["out_bot_context_v6"]).sum()),
            "support_deaths_bot_0_12": int(((g["event_kind"] == "support_death") & g["in_bot_context_v6"]).sum()),
            "adc_deaths_bot_0_12": int(((g["event_kind"] == "adc_death") & g["in_bot_context_v6"]).sum()),
            "team_objectives_out_bot_0_12": int(((g["event_kind"] == "team_objective") & g["out_bot_context_v6"]).sum()),
            "support_building_kills_out_bot_0_12": int(((g["event_kind"] == "support_building_kill") & g["out_bot_context_v6"]).sum()),
            "support_building_kills_bot_0_12": int(((g["event_kind"] == "support_building_kill") & g["in_bot_context_v6"]).sum()),
            "support_plate_destroys_out_bot_0_12": int(((g["event_kind"] == "support_plate_destroy") & g["out_bot_context_v6"]).sum()),
            "support_plate_destroys_bot_0_12": int(((g["event_kind"] == "support_plate_destroy") & g["in_bot_context_v6"]).sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = load_json(Path(args.config))

    rows: List[Dict[str, Any]] = []
    pos_rows: List[Dict[str, Any]] = []
    dirs = list(match_dirs(raw_root, args.limit_matches))
    print(f"[Input] match dirs={len(dirs):,} raw_root={raw_root}")

    if args.workers <= 1:
        for i, match_dir in enumerate(dirs, start=1):
            r, p = process_match(match_dir, args.max_minute)
            rows.extend(r)
            pos_rows.extend(p)
            if i % 5000 == 0 or i == len(dirs):
                print(f"[Scan] {i:,}/{len(dirs):,} matches rows={len(rows):,} pos_events={len(pos_rows):,}")
    else:
        print(f"[Scan] workers={args.workers}")
        chunk_size = max(100, int(np.ceil(len(dirs) / (args.workers * 24))))
        chunks = [dirs[i:i + chunk_size] for i in range(0, len(dirs), chunk_size)]
        print(f"[Scan] chunks={len(chunks):,} chunk_size~={chunk_size:,}")
        scanned = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_match_chunk, chunk, args.max_minute) for chunk in chunks]
            for i, fut in enumerate(as_completed(futures), start=1):
                r, p = fut.result()
                rows.extend(r)
                pos_rows.extend(p)
                scanned += len(chunks[i - 1]) if i - 1 < len(chunks) else 0
                if i % 10 == 0 or i == len(futures):
                    approx = min(len(dirs), i * chunk_size)
                    print(f"[Scan] chunks {i:,}/{len(futures):,} approx_matches={approx:,} rows={len(rows):,} pos_events={len(pos_rows):,}")

    context = pd.DataFrame(rows)
    pos_events = pd.DataFrame(pos_rows)
    pos_events = classify_event_positions(pos_events, config, args.chunk_size)
    pos_agg = aggregate_position_events(pos_events)

    if not pos_agg.empty:
        context = context.merge(pos_agg, on=JOIN_KEYS, how="left")
    for col in [
        "support_kill_assists_out_bot_0_12",
        "support_kill_assists_bot_0_12",
        "support_deaths_out_bot_0_12",
        "support_deaths_bot_0_12",
        "adc_deaths_bot_0_12",
        "team_objectives_out_bot_0_12",
        "support_building_kills_out_bot_0_12",
        "support_building_kills_bot_0_12",
        "support_plate_destroys_out_bot_0_12",
        "support_plate_destroys_bot_0_12",
    ]:
        if col not in context.columns:
            context[col] = 0
        context[col] = context[col].fillna(0).astype(int)

    context["botlane_deaths_bot_0_12"] = context["support_deaths_bot_0_12"] + context["adc_deaths_bot_0_12"]
    context["support_active_events_out_bot_0_12"] = (
        context["support_kill_assists_out_bot_0_12"]
        + context["support_deaths_out_bot_0_12"]
        + context["team_objectives_out_bot_0_12"]
        + context["support_building_kills_out_bot_0_12"]
        + context["support_plate_destroys_out_bot_0_12"]
    )
    context["events_with_position_0_12"] = (
        context["support_kill_assists_out_bot_0_12"]
        + context["support_kill_assists_bot_0_12"]
        + context["support_deaths_out_bot_0_12"]
        + context["support_deaths_bot_0_12"]
        + context["adc_deaths_bot_0_12"]
        + context["team_objectives_out_bot_0_12"]
        + context["support_building_kills_out_bot_0_12"]
        + context["support_building_kills_bot_0_12"]
        + context["support_plate_destroys_out_bot_0_12"]
        + context["support_plate_destroys_bot_0_12"]
    )

    context_path = outdir / args.out_name
    events_path = outdir / "support_event_positions_m12.parquet"
    context.sort_values(JOIN_KEYS).to_parquet(context_path, index=False)
    pos_events.sort_values(JOIN_KEYS + ["minute", "event_kind"]).to_parquet(events_path, index=False)

    summary = {
        "rows": int(len(context)),
        "match_team_keys": int(context[JOIN_KEYS].drop_duplicates().shape[0]),
        "position_events": int(len(pos_events)),
        "raw_root": str(raw_root.resolve()),
        "max_minute": args.max_minute,
        "ward_position_note": "Riot WARD_PLACED/WARD_KILL events in this dataset do not expose map positions, so ward evidence is counted but not spatially classified.",
        "means": {
            col: float(context[col].mean())
            for col in context.columns
            if col.endswith("_0_12") and pd.api.types.is_numeric_dtype(context[col])
        },
    }
    (outdir / "support_event_context_metadata.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[Saved] {context_path}")
    print(f"[Saved] {events_path}")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
