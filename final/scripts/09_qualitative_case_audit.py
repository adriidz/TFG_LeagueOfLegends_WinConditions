#!/usr/bin/env python3
"""
09_qualitative_case_audit.py -- Consolidated qualitative audit for HistGBT cases.

This script replaces the fragmented 09/10/11 workflow for the final report:
it selects top-error and bottom-error cases, joins model predictions, label
components, frame-level positions, Riot timeline events, and map plots.

The goal is evidence, not risky automatic storytelling. The exported notes and
tags help manual inspection of support-ADC separation, geometry, and chaotic
early-game context.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "ProgresoActual2" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from build_geometry_v5_frame_state_distributions import classify_chunk_absolute  # noqa: E402


DEFAULT_TEST = REPO_ROOT / "final" / "data" / "training" / "test.parquet"
DEFAULT_MODEL_DIR = REPO_ROOT / "final" / "models" / "gbt"
DEFAULT_SCORES = REPO_ROOT / "final" / "data" / "scores" / "support_scores_v5_geometry_m12.parquet"
DEFAULT_FRAME_STATE = REPO_ROOT / "final" / "data" / "frame_state" / "support_frame_state.parquet"
DEFAULT_CONFIG = REPO_ROOT / "ProgresoActual2" / "data" / "geometry" / "manual_geometry_v5_config.json"
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw" / "raw" / "europe"
DEFAULT_EXPERT_REFERENCE = REPO_ROOT / "ProgresoActual" / "references" / "manual_support_champion_reference.csv"
DEFAULT_OUTDIR = REPO_ROOT / "final" / "analysis" / "qualitative_case_audit"

TARGET_COL = "support_roam_score"
MISSING_TOKEN = "__MISSING__"
JOIN_KEYS = ["match_id", "team_id"]
BOT_CONTEXT_ZONES = {"BOT_LANE_CORE", "BOT_SIDE_NEAR", "RIVER_BOT", "DRAGON_AREA"}
BASE_ZONES = {"BLUE_BASE", "RED_BASE"}
FAR_ADC_THRESHOLD = 2500.0
WEIGHTS = np.asarray([0.45, 0.35, 0.20], dtype=float)
GAMMA = 0.75
XP_RATIO_MIN = 0.60
XP_RATIO_MAX = 1.00
EARLY_MAX_MINUTE = 12.0
LABEL_START_MINUTE = 5.0
LABEL_MAX_MINUTE = 12.0
SCORE_BINS: List[Tuple[str, float, float]] = [
    ("very_low", 0.0, 0.25),
    ("low_mid", 0.25, 0.50),
    ("high_mid", 0.50, 0.75),
    ("very_high", 0.75, 1.0000001),
]
ROLES = ["top", "jungle", "middle", "bottom", "utility"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Consolidated qualitative case audit.")
    p.add_argument("--test", default=str(DEFAULT_TEST))
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--scores", default=str(DEFAULT_SCORES))
    p.add_argument("--frame-state", default=str(DEFAULT_FRAME_STATE))
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    p.add_argument("--expert-reference", default=str(DEFAULT_EXPERT_REFERENCE))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--bottom-n", type=int, default=20)
    p.add_argument("--start-minute", type=float, default=LABEL_START_MINUTE)
    p.add_argument("--max-minute", type=float, default=LABEL_MAX_MINUTE)
    p.add_argument("--early-max-minute", type=float, default=EARLY_MAX_MINUTE)
    p.add_argument("--chunk-size", type=int, default=500000)
    p.add_argument("--max-note-events", type=int, default=8)
    return p.parse_args()


def encode_features(df: pd.DataFrame, feature_cols: List[str], encoder: Any) -> np.ndarray:
    raw = df[feature_cols].copy()
    for col in feature_cols:
        raw[col] = raw[col].fillna(MISSING_TOKEN).astype(str)
    return encoder.transform(raw)


def add_predictions(test_path: Path, model_dir: Path) -> pd.DataFrame:
    model = joblib.load(model_dir / "gbt_model_raw.joblib")
    preprocess = joblib.load(model_dir / "preprocess.joblib")
    feature_cols: List[str] = preprocess["feature_columns"]
    encoder = preprocess["encoder"]
    df = pd.read_parquet(test_path)
    pred = model.predict(encode_features(df, feature_cols, encoder))
    out = df.copy()
    out["prediction"] = pred
    out["actual"] = out[TARGET_COL].astype(float)
    out["signed_error"] = out["prediction"] - out["actual"]
    out["abs_error"] = out["signed_error"].abs()
    return out


def normalize_champion_name(name: Any) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def load_expert_reference(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    ref = pd.read_csv(path)
    required = {"champion_name", "expert_archetype", "expert_support_roam_score", "expert_confidence"}
    missing = required - set(ref.columns)
    if missing:
        raise ValueError(f"Expert reference missing columns: {sorted(missing)}")
    out = ref.copy()
    out["_champion_key"] = out["champion_name"].map(normalize_champion_name)
    out["expert_support_roam_score"] = pd.to_numeric(out["expert_support_roam_score"], errors="coerce")
    out["expert_confidence"] = pd.to_numeric(out["expert_confidence"], errors="coerce")
    return out


def add_expert_context(cases: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    out = cases.copy()
    if reference.empty:
        out["ally_support_expert_score"] = np.nan
        out["ally_support_expert_archetype"] = ""
        out["ally_support_expert_confidence"] = np.nan
        out["enemy_support_expert_score"] = np.nan
        out["enemy_support_expert_archetype"] = ""
        out["enemy_support_expert_confidence"] = np.nan
        return out

    ref_cols = [
        "_champion_key",
        "expert_support_roam_score",
        "expert_archetype",
        "expert_confidence",
    ]
    ally_ref = reference[ref_cols].rename(
        columns={
            "_champion_key": "_ally_support_key",
            "expert_support_roam_score": "ally_support_expert_score",
            "expert_archetype": "ally_support_expert_archetype",
            "expert_confidence": "ally_support_expert_confidence",
        }
    )
    enemy_ref = reference[ref_cols].rename(
        columns={
            "_champion_key": "_enemy_support_key",
            "expert_support_roam_score": "enemy_support_expert_score",
            "expert_archetype": "enemy_support_expert_archetype",
            "expert_confidence": "enemy_support_expert_confidence",
        }
    )
    out["_ally_support_key"] = out["ally_utility_champion_name"].map(normalize_champion_name)
    out["_enemy_support_key"] = out["enemy_utility_champion_name"].map(normalize_champion_name)
    out = out.merge(ally_ref, on="_ally_support_key", how="left")
    out = out.merge(enemy_ref, on="_enemy_support_key", how="left")
    return out.drop(columns=["_ally_support_key", "_enemy_support_key"], errors="ignore")


def add_empirical_champion_means(cases: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    out = cases.copy()
    required = {"support_champion_name", "support_roam_score_v5_geometry"}
    if not required.issubset(scores.columns):
        out["ally_support_champion_mean_score"] = np.nan
        out["ally_support_champion_n"] = np.nan
        out["enemy_support_champion_mean_score"] = np.nan
        out["enemy_support_champion_n"] = np.nan
        return out

    means = (
        scores.groupby("support_champion_name", dropna=False)["support_roam_score_v5_geometry"]
        .agg(champion_mean_score="mean", champion_n="size")
        .reset_index()
    )
    means["_champion_key"] = means["support_champion_name"].map(normalize_champion_name)
    means = means.drop(columns=["support_champion_name"])

    ally_means = means.rename(
        columns={
            "_champion_key": "_ally_support_key",
            "champion_mean_score": "ally_support_champion_mean_score",
            "champion_n": "ally_support_champion_n",
        }
    )
    enemy_means = means.rename(
        columns={
            "_champion_key": "_enemy_support_key",
            "champion_mean_score": "enemy_support_champion_mean_score",
            "champion_n": "enemy_support_champion_n",
        }
    )
    out["_ally_support_key"] = out["ally_utility_champion_name"].map(normalize_champion_name)
    out["_enemy_support_key"] = out["enemy_utility_champion_name"].map(normalize_champion_name)
    out = out.merge(ally_means, on="_ally_support_key", how="left")
    out = out.merge(enemy_means, on="_enemy_support_key", how="left")
    return out.drop(columns=["_ally_support_key", "_enemy_support_key"], errors="ignore")


def select_cases(predictions: pd.DataFrame, top_n: int, bottom_n: int) -> pd.DataFrame:
    top = predictions.sort_values("abs_error", ascending=False).head(top_n).copy()
    top["case_group"] = "top_error"
    top["case_rank"] = np.arange(1, len(top) + 1)
    top["score_band"] = pd.cut(
        top["actual"],
        bins=[0.0, 0.25, 0.50, 0.75, 1.0000001],
        labels=["very_low", "low_mid", "high_mid", "very_high"],
        include_lowest=True,
    ).astype(str)

    bottom_parts: List[pd.DataFrame] = []
    band_order = {band: i for i, (band, _, _) in enumerate(SCORE_BINS)}
    per_band = max(1, bottom_n // len(SCORE_BINS)) if bottom_n >= len(SCORE_BINS) else 1
    used_index: set = set()
    for band, lo, hi in SCORE_BINS:
        part = predictions[
            (predictions["actual"] >= lo)
            & (predictions["actual"] < hi)
            & ~predictions.index.isin(used_index)
        ].sort_values("abs_error", ascending=True).head(per_band)
        used_index.update(part.index.tolist())
        part = part.copy()
        part["score_band"] = band
        bottom_parts.append(part)

    bottom = pd.concat(bottom_parts, ignore_index=False) if bottom_parts else predictions.head(0).copy()
    if len(bottom) < bottom_n:
        fill = predictions[~predictions.index.isin(used_index)].sort_values("abs_error", ascending=True)
        bottom = pd.concat([bottom, fill.head(bottom_n - len(bottom))], ignore_index=False)
    bottom["_band_order"] = bottom["score_band"].map(band_order).fillna(len(SCORE_BINS)).astype(int)
    bottom = bottom.sort_values(["_band_order", "abs_error"], ascending=[True, True]).head(bottom_n).copy()
    bottom = bottom.drop(columns=["_band_order"])
    bottom["case_group"] = "bottom_error"
    bottom["case_rank"] = np.arange(1, len(bottom) + 1)
    if "score_band" not in bottom:
        bottom["score_band"] = ""

    cases = pd.concat([top, bottom], ignore_index=True)
    cases["case_id"] = cases.apply(
        lambda r: f"{str(r['case_group'])}_{int(r['case_rank']):02d}_{r['match_id']}_{int(r['team_id'])}",
        axis=1,
    )
    return cases


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def event_minute(event: Dict[str, Any]) -> float:
    return float(event.get("timestamp", 0.0)) / 60000.0


def participant_maps(match: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], Dict[Tuple[int, str], int]]:
    by_pid: Dict[int, Dict[str, Any]] = {}
    role_to_pid: Dict[Tuple[int, str], int] = {}
    for p in (match.get("info") or {}).get("participants") or []:
        pid = p.get("participantId")
        if not isinstance(pid, int):
            continue
        by_pid[pid] = p
        team_id = p.get("teamId")
        role = str(p.get("teamPosition") or "").upper()
        if isinstance(team_id, int) and role:
            role_to_pid[(team_id, role)] = pid
    return by_pid, role_to_pid


def pid_label(pid: Optional[int], by_pid: Dict[int, Dict[str, Any]]) -> str:
    if not isinstance(pid, int) or pid <= 0:
        return ""
    p = by_pid.get(pid, {})
    champion = str(p.get("championName") or f"pid{pid}")
    role = str(p.get("teamPosition") or "")
    return f"{champion}({role},T{p.get('teamId', '')},pid{pid})"


def relation(pid: Optional[int], team_id: int, support_pid: int, adc_pid: int, by_pid: Dict[int, Dict[str, Any]]) -> str:
    if not isinstance(pid, int) or pid <= 0:
        return ""
    if pid == support_pid:
        return "ally_support"
    if pid == adc_pid:
        return "ally_adc"
    p = by_pid.get(pid, {})
    role = str(p.get("teamPosition") or "").lower()
    return f"ally_{role}" if p.get("teamId") == team_id else f"enemy_{role}"


def iter_events(timeline: Dict[str, Any], max_minute: float) -> List[Dict[str, Any]]:
    frames = (timeline.get("info") or {}).get("frames") or timeline.get("frames") or []
    out: List[Dict[str, Any]] = []
    for frame in frames:
        for event in frame.get("events") or []:
            if event_minute(event) <= max_minute:
                out.append(event)
    return sorted(out, key=lambda e: e.get("timestamp", 0))


def kda(events: Iterable[Dict[str, Any]], pid: Optional[int]) -> Dict[str, int]:
    kills = deaths = assists = 0
    if not isinstance(pid, int):
        return {"kills": 0, "deaths": 0, "assists": 0}
    for e in events:
        if e.get("type") != "CHAMPION_KILL":
            continue
        kills += int(e.get("killerId") == pid)
        deaths += int(e.get("victimId") == pid)
        assists += int(pid in (e.get("assistingParticipantIds") or []))
    return {"kills": kills, "deaths": deaths, "assists": assists}


def event_row(
    case: pd.Series,
    event: Dict[str, Any],
    by_pid: Dict[int, Dict[str, Any]],
    support_pid: int,
    adc_pid: int,
) -> Optional[Dict[str, Any]]:
    typ = event.get("type")
    if typ not in {"CHAMPION_KILL", "ELITE_MONSTER_KILL", "BUILDING_KILL", "TURRET_PLATE_DESTROYED"}:
        return None
    team_id = int(case["team_id"])
    killer = event.get("killerId")
    victim = event.get("victimId")
    assists = [int(x) for x in event.get("assistingParticipantIds") or [] if isinstance(x, int)]
    involved = set([pid for pid in [killer, victim] if isinstance(pid, int)]) | set(assists)
    pos = event.get("position") or {}
    return {
        "case_id": case["case_id"],
        "case_group": case["case_group"],
        "case_rank": int(case["case_rank"]),
        "match_id": case["match_id"],
        "team_id": team_id,
        "minute": event_minute(event),
        "event_type": typ,
        "killer": pid_label(killer, by_pid),
        "victim": pid_label(victim, by_pid),
        "assists": "; ".join(pid_label(pid, by_pid) for pid in assists),
        "killer_relation": relation(killer, team_id, support_pid, adc_pid, by_pid),
        "victim_relation": relation(victim, team_id, support_pid, adc_pid, by_pid),
        "assist_relations": "; ".join(relation(pid, team_id, support_pid, adc_pid, by_pid) for pid in assists),
        "monster_type": event.get("monsterType"),
        "building_type": event.get("buildingType"),
        "lane_type": event.get("laneType"),
        "ally_support_involved": support_pid in involved,
        "ally_adc_involved": adc_pid in involved,
        "ally_support_died": victim == support_pid,
        "ally_adc_died": victim == adc_pid,
        "ally_support_assist": support_pid in assists,
        "ally_adc_assist": adc_pid in assists,
        "x": pos.get("x"),
        "y": pos.get("y"),
    }


def zone_order(config: Dict[str, Any]) -> List[str]:
    order = ["OUT_OF_MAP", "UNCLASSIFIED"] + list(config["colors"].keys())
    for zone in config["priority"]:
        if zone not in order:
            order.append(zone)
    return order


def classify_xy(x: np.ndarray, y: np.ndarray, config: Dict[str, Any], chunk_size: int) -> np.ndarray:
    order = zone_order(config)
    zone_to_id = {zone: idx for idx, zone in enumerate(order)}
    id_to_zone = np.asarray(order, dtype=object)
    out = np.empty(x.shape[0], dtype=np.int16)
    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        out[start:end] = classify_chunk_absolute(x[start:end], y[start:end], config, zone_to_id)
    return id_to_zone[out]


def load_case_frames(frame_state_path: Path, cases: pd.DataFrame, start_minute: float, max_minute: float) -> pd.DataFrame:
    columns = [
        "match_id", "team_id", "side", "patch", "frame_idx", "minute",
        "support_champion_name", "adc_champion_name", "support_alive", "adc_alive",
        "support_x", "support_y", "adc_x", "adc_y", "support_zone", "adc_zone",
        "support_in_base", "adc_in_base", "support_in_bot_extended", "dist_to_adc",
        "support_xp", "adc_xp",
    ]
    match_ids = set(cases["match_id"].astype(str))
    team_pairs = set(zip(cases["match_id"].astype(str), cases["team_id"].astype(int)))
    df = pd.read_parquet(frame_state_path, columns=columns)
    df = df[
        df["match_id"].astype(str).isin(match_ids)
        & df["minute"].between(start_minute, max_minute, inclusive="both")
    ].copy()
    df = df[df.apply(lambda r: (str(r["match_id"]), int(r["team_id"])) in team_pairs, axis=1)].copy()
    case_lookup = cases[["case_id", "case_group", "case_rank", "match_id", "team_id"]].copy()
    df = df.merge(case_lookup, on=["match_id", "team_id"], how="left")
    return df.sort_values(["case_group", "case_rank", "frame_idx"]).reset_index(drop=True)


def add_v5_flags(frames: pd.DataFrame, config: Dict[str, Any], chunk_size: int) -> pd.DataFrame:
    out = frames.copy()
    out["support_zone_v5_abs"] = classify_xy(out["support_x"].to_numpy(float), out["support_y"].to_numpy(float), config, chunk_size)
    out["adc_zone_v5_abs"] = classify_xy(out["adc_x"].to_numpy(float), out["adc_y"].to_numpy(float), config, chunk_size)
    out["support_in_base_v5"] = out["support_zone_v5_abs"].isin(BASE_ZONES)
    out["adc_in_base_v5"] = out["adc_zone_v5_abs"].isin(BASE_ZONES)
    out["support_in_bot_context_v5"] = out["support_zone_v5_abs"].isin(BOT_CONTEXT_ZONES)
    out["valid_support_frame_v5"] = (
        out["support_alive"].fillna(False)
        & out["support_x"].notna()
        & out["support_y"].notna()
        & ~out["support_in_base_v5"].fillna(False)
    )
    out["valid_coop_frame_v5"] = (
        out["valid_support_frame_v5"]
        & out["adc_alive"].fillna(False)
        & out["adc_x"].notna()
        & out["adc_y"].notna()
        & ~out["adc_in_base_v5"].fillna(False)
    )
    out["out_bot_context_v5"] = out["valid_support_frame_v5"] & ~out["support_in_bot_context_v5"].fillna(False)
    out["far_from_adc_v5"] = out["valid_coop_frame_v5"] & (out["dist_to_adc"].fillna(-1.0) >= FAR_ADC_THRESHOLD)
    out["support_dead_or_base"] = (~out["support_alive"].fillna(False)) | out["support_in_base_v5"].fillna(False)
    out["adc_dead_or_base"] = (~out["adc_alive"].fillna(False)) | out["adc_in_base_v5"].fillna(False)
    out["xp_ratio_frame"] = np.where(out["adc_xp"].fillna(0) > 0, out["support_xp"] / out["adc_xp"], np.nan)
    return out


def xp_gap_from_last(group: pd.DataFrame) -> Tuple[float, float]:
    ordered = group.sort_values("frame_idx")
    support_xp = float(ordered["support_xp"].iloc[-1]) if pd.notna(ordered["support_xp"].iloc[-1]) else np.nan
    adc_xp = float(ordered["adc_xp"].iloc[-1]) if pd.notna(ordered["adc_xp"].iloc[-1]) else np.nan
    if not np.isfinite(adc_xp) or adc_xp <= 0 or not np.isfinite(support_xp):
        return np.nan, np.nan
    ratio = support_xp / adc_xp
    clipped = min(max(ratio, XP_RATIO_MIN), XP_RATIO_MAX)
    return ratio, 1.0 - ((clipped - XP_RATIO_MIN) / (XP_RATIO_MAX - XP_RATIO_MIN))


def reconstruct_scores(frames: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for case_id, group in frames.groupby("case_id", dropna=False):
        support_valid = group[group["valid_support_frame_v5"]]
        coop_valid = group[group["valid_coop_frame_v5"]]
        ratio, xp_gap = xp_gap_from_last(group)
        outside_ratio = float(support_valid["out_bot_context_v5"].mean()) if len(support_valid) else np.nan
        far_ratio = float(coop_valid["far_from_adc_v5"].mean()) if len(coop_valid) else np.nan
        components = np.asarray([outside_ratio, far_ratio, xp_gap], dtype=float)
        valid = np.isfinite(components)
        denom = float((WEIGHTS * valid.astype(float)).sum())
        raw = float((np.nan_to_num(components) * WEIGHTS).sum() / denom) if denom > 0 else np.nan
        score = float(np.clip(raw, 0, 1) ** GAMMA) if np.isfinite(raw) else np.nan
        rows.append({
            "case_id": case_id,
            "frames_in_window": int(len(group)),
            "valid_support_frames_reconstructed": int(len(support_valid)),
            "valid_coop_frames_reconstructed": int(len(coop_valid)),
            "support_dead_or_base_frames": int(group["support_dead_or_base"].sum()),
            "adc_dead_or_base_frames": int(group["adc_dead_or_base"].sum()),
            "outside_ratio_reconstructed": outside_ratio,
            "far_ratio_reconstructed": far_ratio,
            "xp_ratio_reconstructed": ratio,
            "xp_gap_reconstructed": xp_gap,
            "raw_score_reconstructed": raw,
            "score_reconstructed": score,
            "mean_distance_to_adc_reconstructed": float(coop_valid["dist_to_adc"].mean()) if len(coop_valid) else np.nan,
            "max_distance_to_adc": float(group["dist_to_adc"].max()) if group["dist_to_adc"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def draft_string(row: pd.Series) -> str:
    ally = "/".join(str(row.get(f"ally_{role}_champion_name", "")) for role in ROLES)
    enemy = "/".join(str(row.get(f"enemy_{role}_champion_name", "")) for role in ROLES)
    return f"{ally} vs {enemy}"


def evidence_tag(row: pd.Series) -> str:
    if bool(row.get("raw_missing", False)):
        return "raw_missing"
    if row.get("valid_support_frames_v5", 0) < 4 or row.get("valid_coop_frames_v5", 0) < 3:
        return "label_quality_caution"
    botlane_deaths = row.get("support_early_deaths", 0) + row.get("adc_early_deaths", 0)
    if botlane_deaths >= 5 or row.get("bot_related_events_0_12", 0) >= 14:
        return "chaotic_early_game"
    if row.get("actual", 0) >= 0.75 and row.get("outside_ratio_v5", 0) >= 0.70 and row.get("far_ratio_v5", 0) >= 0.70:
        return "clean_roam_like_candidate"
    if row.get("case_group") == "bottom_error":
        actual = float(row.get("actual", 0))
        if actual < 0.25:
            return "accurate_low"
        if actual < 0.75:
            return "accurate_mid"
        return "accurate_high"
    return "manual_review"


def plot_case_map(case: pd.Series, frames: pd.DataFrame, config: Dict[str, Any], outpath: Path) -> None:
    map_max = float(config.get("map_max", 14800.0))
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for zone, points in (config.get("polygons") or {}).items():
        color = (config.get("colors") or {}).get(zone, "#dddddd")
        patch = Polygon(points, closed=True, facecolor=color, edgecolor="white", alpha=0.18, linewidth=0.7)
        ax.add_patch(patch)

    ordered = frames.sort_values("minute")
    support_valid = ordered["valid_support_frame_v5"].fillna(False)
    adc_valid = ordered["valid_coop_frame_v5"].fillna(False)
    ax.scatter(
        ordered.loc[~adc_valid, "adc_x"],
        ordered.loc[~adc_valid, "adc_y"],
        color="#9e9e9e",
        marker="s",
        s=46,
        alpha=0.75,
        label="ADC not counted",
    )
    ax.scatter(
        ordered.loc[adc_valid, "adc_x"],
        ordered.loc[adc_valid, "adc_y"],
        color="#1f77b4",
        marker="s",
        s=48,
        label="ADC counted",
    )
    ax.scatter(
        ordered.loc[~support_valid, "support_x"],
        ordered.loc[~support_valid, "support_y"],
        color="#9e9e9e",
        marker="o",
        s=46,
        alpha=0.75,
        label="Support not counted",
    )
    ax.scatter(
        ordered.loc[support_valid, "support_x"],
        ordered.loc[support_valid, "support_y"],
        color="#d62728",
        marker="o",
        s=48,
        label="Support counted",
    )
    for _, r in ordered.iterrows():
        minute = int(round(float(r["minute"])))
        support_color = "#8b0000" if bool(r["valid_support_frame_v5"]) else "#666666"
        adc_color = "#003f7f" if bool(r["valid_coop_frame_v5"]) else "#666666"
        ax.text(r["support_x"], r["support_y"], f"S{minute}", fontsize=7, color=support_color)
        ax.text(r["adc_x"], r["adc_y"], f"A{minute}", fontsize=7, color=adc_color)

    ally_bot = f"{case.get('ally_utility_champion_name', '')} + {case.get('ally_bottom_champion_name', '')}"
    enemy_bot = f"{case.get('enemy_utility_champion_name', '')} + {case.get('enemy_bottom_champion_name', '')}"
    ally_expert = case.get("ally_support_expert_score")
    enemy_expert = case.get("enemy_support_expert_score")
    ally_expert_txt = "NA" if pd.isna(ally_expert) else f"{float(ally_expert):.2f}"
    enemy_expert_txt = "NA" if pd.isna(enemy_expert) else f"{float(enemy_expert):.2f}"
    ally_arch = str(case.get("ally_support_expert_archetype", "") or "")
    enemy_arch = str(case.get("enemy_support_expert_archetype", "") or "")
    ally_mean = case.get("ally_support_champion_mean_score")
    enemy_mean = case.get("enemy_support_champion_mean_score")
    ally_n = case.get("ally_support_champion_n")
    enemy_n = case.get("enemy_support_champion_n")
    ally_mean_txt = "NA" if pd.isna(ally_mean) else f"{float(ally_mean):.2f}"
    enemy_mean_txt = "NA" if pd.isna(enemy_mean) else f"{float(enemy_mean):.2f}"
    ally_n_txt = "" if pd.isna(ally_n) else f", n={int(ally_n)}"
    enemy_n_txt = "" if pd.isna(enemy_n) else f", n={int(enemy_n)}"
    title = (
        f"{case['case_group']} #{int(case['case_rank'])}: {case['match_id']} T{int(case['team_id'])} "
        f"({case.get('side', '')}, patch {case.get('patch', '')})\n"
        f"Ally bot: {ally_bot}  vs  Enemy bot: {enemy_bot}\n"
        f"Expert support roam: ally={ally_expert_txt} ({ally_arch})  enemy={enemy_expert_txt} ({enemy_arch})\n"
        f"Champion mean score: ally={ally_mean_txt}{ally_n_txt}  enemy={enemy_mean_txt}{enemy_n_txt}\n"
        f"pred={case['prediction']:.3f} actual={case['actual']:.3f} abs_err={case['abs_error']:.3f} "
        f"tag={case.get('evidence_tag', '')}"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, map_max)
    ax.set_ylim(0, map_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_case_timeline(case: pd.Series, frames: pd.DataFrame, outpath: Path) -> None:
    ordered = frames.sort_values("minute")
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(ordered["minute"], ordered["dist_to_adc"], marker="o", color="#333333")
    axes[0].axhline(FAR_ADC_THRESHOLD, color="#d62728", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Dist support-ADC")
    axes[1].plot(ordered["minute"], ordered["out_bot_context_v5"].astype(int), marker="o", label="out bot")
    axes[1].plot(ordered["minute"], ordered["far_from_adc_v5"].astype(int), marker="s", label="far adc")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_ylabel("Flags")
    axes[1].legend(fontsize=8)
    axes[2].plot(ordered["minute"], ordered["support_xp"], marker="o", label="support XP")
    axes[2].plot(ordered["minute"], ordered["adc_xp"], marker="s", label="ADC XP")
    axes[2].set_ylabel("XP")
    axes[2].set_xlabel("Minute")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"{case['case_id']} frame-level label audit", fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            display[col] = display[col].fillna("").astype(str)
    headers = list(display.columns)
    rows = display.astype(str).values.tolist()
    if not rows:
        return ""
    widths = [max(len(str(h)), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    lines = [
        "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows)
    return "\n".join(lines)


def event_lines(events: pd.DataFrame, max_events: int) -> List[str]:
    if events.empty:
        return ["- No early timeline events exported for this case."]
    rows = []
    focused = events[
        events["ally_support_involved"].fillna(False)
        | events["ally_adc_involved"].fillna(False)
        | events["ally_support_died"].fillna(False)
        | events["ally_adc_died"].fillna(False)
    ].sort_values("minute")
    if focused.empty:
        focused = events.sort_values("minute")
    for _, e in focused.head(max_events).iterrows():
        desc = f"- min {e['minute']:.2f}: {e['event_type']}"
        if e.get("killer"):
            desc += f" | {e.get('killer', '')} -> {e.get('victim', '')}"
        if e.get("assists"):
            desc += f" | assists: {e.get('assists')}"
        flags = []
        if e.get("ally_support_died"):
            flags.append("support died")
        if e.get("ally_adc_died"):
            flags.append("ADC died")
        if e.get("ally_support_assist"):
            flags.append("support assist")
        if e.get("ally_adc_assist"):
            flags.append("ADC assist")
        if flags:
            desc += f" ({'; '.join(flags)})"
        rows.append(desc)
    return rows


def expert_text(score: Any, archetype: Any) -> str:
    if pd.isna(score):
        return "NA"
    arch = str(archetype or "")
    return f"{float(score):.3f} ({arch})" if arch and arch != "nan" else f"{float(score):.3f}"


def mean_text(score: Any, n: Any) -> str:
    if pd.isna(score):
        return "NA"
    n_txt = "" if pd.isna(n) else f", n={int(n)}"
    return f"{float(score):.3f}{n_txt}"


def build_notes(case_index: pd.DataFrame, events: pd.DataFrame, outdir: Path, max_events: int) -> str:
    cols = [
        "case_group", "case_rank", "match_id", "team_id", "side", "patch",
        "ally_utility_champion_name", "ally_bottom_champion_name",
        "enemy_utility_champion_name", "enemy_bottom_champion_name",
        "ally_support_expert_score", "ally_support_expert_archetype",
        "ally_support_champion_mean_score",
        "prediction", "actual", "abs_error", "outside_ratio_v5", "far_ratio_v5",
        "support_early_deaths", "adc_early_deaths", "evidence_tag",
    ]
    parts = [
        "# Qualitative Case Audit",
        "",
        "This report consolidates model errors, label components, frame-level positions, map plots, and raw Riot timeline evidence.",
        "",
        "## Top Errors: donde falla el modelo",
        "",
        markdown_table(case_index[case_index["case_group"] == "top_error"][[c for c in cols if c in case_index.columns]]),
        "",
        "## Bottom Errors: cuando el modelo acierta",
        "",
        markdown_table(case_index[case_index["case_group"] == "bottom_error"][[c for c in cols if c in case_index.columns]]),
        "",
        "## Patrones encontrados",
        "",
    ]
    tag_counts = case_index["evidence_tag"].value_counts(dropna=False).reset_index()
    tag_counts.columns = ["evidence_tag", "cases"]
    parts.append(markdown_table(tag_counts))
    parts.extend([
        "",
        "## Limitaciones de etiqueta",
        "",
        "- `support_roam_score` debe leerse como `roam-like displacement` o separacion support-ADC, no como intencion tactica garantizada.",
        "- Casos con muchas muertes tempranas o pocos frames validos deben usarse como cautela metodologica.",
        "- Los mapas cronologicos en `case_plots/` permiten auditar si las posiciones y zonas geometricas parecen correctas.",
        "",
        "## Casos recomendados para la memoria",
        "",
    ])
    top_examples = case_index[case_index["case_group"] == "top_error"].sort_values("abs_error", ascending=False).head(5)
    bottom_examples = case_index[case_index["case_group"] == "bottom_error"].sort_values(
        ["actual", "abs_error"], ascending=[False, True]
    ).head(3)
    preferred = pd.concat([top_examples, bottom_examples], ignore_index=True)
    for _, case in preferred.iterrows():
        case_events = events[events["case_id"] == case["case_id"]]
        map_rel = f"case_plots/{case['case_id']}_map.png"
        timeline_rel = f"case_plots/{case['case_id']}_timeline.png"
        parts.extend([
            f"### {case['case_group']} #{int(case['case_rank'])}: {case['match_id']} T{int(case['team_id'])}",
            "",
            f"Draft: {draft_string(case)}.",
            (
                f"Expert expected support score: ally {case.get('ally_utility_champion_name')}="
                f"{expert_text(case.get('ally_support_expert_score', np.nan), case.get('ally_support_expert_archetype', ''))}, "
                f"enemy {case.get('enemy_utility_champion_name')}="
                f"{expert_text(case.get('enemy_support_expert_score', np.nan), case.get('enemy_support_expert_archetype', ''))}."
            ),
            (
                f"Empirical champion mean: ally {case.get('ally_utility_champion_name')}="
                f"{mean_text(case.get('ally_support_champion_mean_score', np.nan), case.get('ally_support_champion_n', np.nan))}, "
                f"enemy {case.get('enemy_utility_champion_name')}="
                f"{mean_text(case.get('enemy_support_champion_mean_score', np.nan), case.get('enemy_support_champion_n', np.nan))}."
            ),
            f"Prediccion={case['prediction']:.3f}, actual={case['actual']:.3f}, abs_error={case['abs_error']:.3f}, tag={case['evidence_tag']}.",
            f"Mapa: `{map_rel}`. Timeline frame-level: `{timeline_rel}`.",
            "",
            "Eventos tempranos relevantes:",
            *event_lines(case_events, max_events),
            "",
        ])
    (outdir / "case_notes.md").write_text("\n".join(parts), encoding="utf-8")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    plots_dir = outdir / "case_plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    for path in [
        outdir / "case_index.csv",
        outdir / "case_event_timeline.csv",
        outdir / "case_frame_timeline.csv",
        outdir / "case_notes.md",
        outdir / "metadata.json",
    ]:
        if path.exists():
            path.unlink()
    for path in plots_dir.glob("*.png"):
        path.unlink()

    config = load_json(Path(args.config))
    predictions = add_predictions(Path(args.test), Path(args.model_dir))
    cases = select_cases(predictions, args.top_n, args.bottom_n)
    expert_reference = load_expert_reference(Path(args.expert_reference))
    cases = add_expert_context(cases, expert_reference)

    scores = pd.read_parquet(args.scores)
    cases = add_empirical_champion_means(cases, scores)
    score_cols = [
        "match_id", "team_id", "valid_support_frames_v5", "valid_coop_frames_v5",
        "outside_ratio_v5", "far_ratio_v5", "xp_gap_v5", "frames_out_bot_context_v5",
        "frames_in_bot_context_v5", "frames_far_from_adc_v5", "mean_distance_to_adc_v5",
        "support_adc_xp_ratio_v5", "support_score_confidence_v5",
        "raw_support_roam_score_v5_geometry", "support_roam_score_v5_geometry",
    ]
    duplicate_score_cols = [c for c in score_cols if c not in JOIN_KEYS and c in cases.columns]
    if duplicate_score_cols:
        cases = cases.drop(columns=duplicate_score_cols)
    cases = cases.merge(scores[score_cols], on=JOIN_KEYS, how="left")

    frames = load_case_frames(Path(args.frame_state), cases, args.start_minute, args.max_minute)
    frames = add_v5_flags(frames, config, args.chunk_size)
    reconstructed = reconstruct_scores(frames)
    cases = cases.merge(reconstructed, on="case_id", how="left")
    cases["score_reconstructed_delta"] = cases["score_reconstructed"] - cases["support_roam_score_v5_geometry"]
    cases["raw_score_reconstructed_delta"] = cases["raw_score_reconstructed"] - cases["raw_support_roam_score_v5_geometry"]

    event_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for _, case in cases.iterrows():
        match_id = str(case["match_id"])
        team_id = int(case["team_id"])
        match_path = Path(args.raw_root) / match_id / "match.json"
        timeline_path = Path(args.raw_root) / match_id / "timeline.json"
        raw_missing = not match_path.exists() or not timeline_path.exists()
        support_pid = adc_pid = enemy_support_pid = enemy_adc_pid = None
        by_pid: Dict[int, Dict[str, Any]] = {}
        early_events: List[Dict[str, Any]] = []
        kill_events: List[Dict[str, Any]] = []
        support_final = adc_final = {}
        if not raw_missing:
            match = load_json(match_path)
            timeline = load_json(timeline_path)
            by_pid, role_to_pid = participant_maps(match)
            support_pid = role_to_pid.get((team_id, "UTILITY"))
            adc_pid = role_to_pid.get((team_id, "BOTTOM"))
            enemy_team = 200 if team_id == 100 else 100
            enemy_support_pid = role_to_pid.get((enemy_team, "UTILITY"))
            enemy_adc_pid = role_to_pid.get((enemy_team, "BOTTOM"))
            early_events = iter_events(timeline, args.early_max_minute)
            kill_events = [e for e in early_events if e.get("type") == "CHAMPION_KILL"]
            if isinstance(support_pid, int) and isinstance(adc_pid, int):
                support_final = by_pid.get(support_pid, {})
                adc_final = by_pid.get(adc_pid, {})
                for event in early_events:
                    row = event_row(case, event, by_pid, support_pid, adc_pid)
                    if row is not None:
                        event_rows.append(row)

        support_kda = kda(kill_events, support_pid)
        adc_kda = kda(kill_events, adc_pid)
        enemy_support_kda = kda(kill_events, enemy_support_pid)
        enemy_adc_kda = kda(kill_events, enemy_adc_pid)
        bot_pids = {pid for pid in [support_pid, adc_pid, enemy_support_pid, enemy_adc_pid] if isinstance(pid, int)}
        bot_related = 0
        for e in kill_events:
            involved = set([e.get("killerId"), e.get("victimId")]) | set(e.get("assistingParticipantIds") or [])
            bot_related += int(bool(involved & bot_pids))

        row = case.to_dict()
        row.update({
            "raw_missing": raw_missing,
            "support_early_kills": support_kda["kills"],
            "support_early_deaths": support_kda["deaths"],
            "support_early_assists": support_kda["assists"],
            "adc_early_kills": adc_kda["kills"],
            "adc_early_deaths": adc_kda["deaths"],
            "adc_early_assists": adc_kda["assists"],
            "enemy_support_early_kda": f"{enemy_support_kda['kills']}/{enemy_support_kda['deaths']}/{enemy_support_kda['assists']}",
            "enemy_adc_early_kda": f"{enemy_adc_kda['kills']}/{enemy_adc_kda['deaths']}/{enemy_adc_kda['assists']}",
            "bot_related_events_0_12": bot_related,
            "support_final_kda": f"{support_final.get('kills')}/{support_final.get('deaths')}/{support_final.get('assists')}" if support_final else "",
            "adc_final_kda": f"{adc_final.get('kills')}/{adc_final.get('deaths')}/{adc_final.get('assists')}" if adc_final else "",
            "support_final_vision": support_final.get("visionScore") if support_final else np.nan,
            "support_final_gold": support_final.get("goldEarned") if support_final else np.nan,
            "adc_final_gold": adc_final.get("goldEarned") if adc_final else np.nan,
        })
        summary_rows.append(row)

    case_index = pd.DataFrame(summary_rows)
    if not case_index.empty:
        case_index["evidence_tag"] = case_index.apply(evidence_tag, axis=1)
    events = pd.DataFrame(event_rows)

    plot_paths = []
    for _, case in case_index.iterrows():
        case_frames = frames[frames["case_id"] == case["case_id"]].copy()
        if case_frames.empty:
            continue
        map_path = plots_dir / f"{case['case_id']}_map.png"
        timeline_path = plots_dir / f"{case['case_id']}_timeline.png"
        plot_case_map(case, case_frames, config, map_path)
        plot_case_timeline(case, case_frames, timeline_path)
        plot_paths.append(str(map_path))

    frame_cols_first = [
        "case_id", "case_group", "case_rank", "match_id", "team_id", "side", "patch",
        "frame_idx", "minute", "support_champion_name", "adc_champion_name",
        "support_alive", "adc_alive", "support_x", "support_y", "adc_x", "adc_y",
        "support_zone_v5_abs", "adc_zone_v5_abs", "dist_to_adc",
        "valid_support_frame_v5", "valid_coop_frame_v5", "out_bot_context_v5", "far_from_adc_v5",
        "support_xp", "adc_xp", "xp_ratio_frame",
    ]
    frames[[c for c in frame_cols_first if c in frames.columns]].to_csv(outdir / "case_frame_timeline.csv", index=False)
    events.to_csv(outdir / "case_event_timeline.csv", index=False)

    preferred_cols = [
        "case_id", "case_group", "case_rank", "score_band", "match_id", "team_id", "side", "patch",
        "prediction", "actual", "signed_error", "abs_error",
        "ally_top_champion_name", "ally_jungle_champion_name", "ally_middle_champion_name",
        "ally_bottom_champion_name", "ally_utility_champion_name",
        "enemy_top_champion_name", "enemy_jungle_champion_name", "enemy_middle_champion_name",
        "enemy_bottom_champion_name", "enemy_utility_champion_name",
        "ally_support_expert_score", "ally_support_expert_archetype", "ally_support_expert_confidence",
        "enemy_support_expert_score", "enemy_support_expert_archetype", "enemy_support_expert_confidence",
        "ally_support_champion_mean_score", "ally_support_champion_n",
        "enemy_support_champion_mean_score", "enemy_support_champion_n",
        "valid_support_frames_v5", "valid_coop_frames_v5", "outside_ratio_v5", "far_ratio_v5",
        "xp_gap_v5", "support_score_confidence_v5", "score_reconstructed_delta",
        "raw_score_reconstructed_delta", "support_early_kills", "support_early_deaths",
        "support_early_assists", "adc_early_kills", "adc_early_deaths", "adc_early_assists",
        "bot_related_events_0_12", "support_final_kda", "adc_final_kda", "evidence_tag", "raw_missing",
    ]
    case_index[[c for c in preferred_cols if c in case_index.columns]].to_csv(outdir / "case_index.csv", index=False)
    build_notes(case_index, events, outdir, args.max_note_events)

    max_delta = float(np.nanmax(np.abs(case_index["score_reconstructed_delta"]))) if len(case_index) else np.nan
    max_raw_delta = float(np.nanmax(np.abs(case_index["raw_score_reconstructed_delta"]))) if len(case_index) else np.nan
    meta = {
        "test_path": str(Path(args.test).resolve()),
        "model_dir": str(Path(args.model_dir).resolve()),
        "scores_path": str(Path(args.scores).resolve()),
        "frame_state_path": str(Path(args.frame_state).resolve()),
        "expert_reference_path": str(Path(args.expert_reference).resolve()),
        "raw_root": str(Path(args.raw_root).resolve()),
        "outdir": str(outdir.resolve()),
        "top_n": args.top_n,
        "bottom_n": args.bottom_n,
        "cases": int(len(case_index)),
        "top_error_cases": int((case_index["case_group"] == "top_error").sum()) if len(case_index) else 0,
        "bottom_error_cases": int((case_index["case_group"] == "bottom_error").sum()) if len(case_index) else 0,
        "raw_missing_cases": int(case_index["raw_missing"].sum()) if len(case_index) else 0,
        "frame_rows": int(len(frames)),
        "event_rows": int(len(events)),
        "map_plots": int(len(plot_paths)),
        "timeline_plots": int(len(plot_paths)),
        "expert_reference_rows": int(len(expert_reference)),
        "max_score_reconstruction_delta": max_delta,
        "max_raw_score_reconstruction_delta": max_raw_delta,
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Cases] {meta['cases']} total ({meta['top_error_cases']} top, {meta['bottom_error_cases']} bottom)")
    print(f"[Raw] missing={meta['raw_missing_cases']} events={meta['event_rows']}")
    print(f"[Frames] rows={meta['frame_rows']} max_delta={max_delta:.10f}")
    print(f"[Plots] maps={meta['map_plots']} timelines={meta['timeline_plots']}")
    print(f"[Saved] {outdir.resolve()}")


if __name__ == "__main__":
    main()
