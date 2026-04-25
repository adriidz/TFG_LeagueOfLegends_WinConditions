#!/usr/bin/env python3
"""
Score many support_v2 configurations from a cached support frame-state parquet.

This script avoids re-reading raw matches and lets you compare:
- start_minute
- max_minute
- far_adc_threshold
- formula weights

Outputs
-------
1) Long parquet: one row per (match_id, team_id, config_id)
2) Config summary parquet/csv with coverage and distribution stats
3) Optional per-config champion summaries
4) Optional canonical support_scores parquet for one selected config
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

DEFAULT_FRAME_STATE_DIR = os.path.join("ProgresoActual", "data", "clean", "frame_state")
DEFAULT_OUT_DIR = os.path.join("ProgresoActual", "analysis", "support_grid")
DEFAULT_FRAME_STATE_NAME = "support_frame_state"

JOIN_KEYS = ["match_id", "team_id"]


def format_sample_suffix(sample_frac: Optional[float]) -> str:
    if sample_frac is None or sample_frac <= 0.0 or sample_frac >= 1.0:
        return ""
    return f"_sample{int(round(sample_frac * 100))}"


def format_window_tag(max_minute: float) -> str:
    return f"m{int(round(max_minute)):02d}"


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_weight_triplets(values: List[str]) -> List[tuple[float, float, float]]:
    out: List[tuple[float, float, float]] = []
    for item in values:
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 3:
            raise SystemExit(f"Peso inválido '{item}'. Usa formato a,b,c")
        try:
            triplet = tuple(float(x) for x in parts)
        except Exception as exc:
            raise SystemExit(f"Peso inválido '{item}': {exc}")
        if sum(triplet) <= 0:
            raise SystemExit(f"La suma de pesos debe ser > 0: {item}")
        out.append(triplet)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search support score configs from support_frame_state parquet.")
    p.add_argument("--frame-state-dir", default=DEFAULT_FRAME_STATE_DIR)
    p.add_argument("--frame-state-name", default=DEFAULT_FRAME_STATE_NAME)
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--outdir", default=DEFAULT_OUT_DIR)
    p.add_argument("--start-minutes", nargs="+", type=float, required=True)
    p.add_argument("--max-minutes", nargs="+", type=float, required=True)
    p.add_argument("--far-adc-thresholds", nargs="+", type=float, default=[2500.0])
    p.add_argument(
        "--weight-triplets",
        nargs="+",
        default=["0.45,0.35,0.20"],
        help="Triples w_outside,w_far,w_xp. Example: 0.35,0.40,0.25 0.30,0.45,0.25",
    )
    p.add_argument("--min-support-frames", type=int, default=2)
    p.add_argument("--xp-ratio-min", type=float, default=0.60)
    p.add_argument("--xp-ratio-max", type=float, default=1.00)
    p.add_argument("--champion-summary", action="store_true")
    p.add_argument(
        "--export-config-id",
        default=None,
        help="If set, export this config_id as a canonical support_scores parquet.",
    )
    p.add_argument(
        "--export-best",
        choices=["coverage", "score_std", "score_iqr"],
        default=None,
        help="Export the best config according to the selected summary criterion.",
    )
    p.add_argument(
        "--export-support-scores-path",
        default=None,
        help=(
            "Output path for the selected canonical support_scores parquet. "
            "Defaults to ProgresoActual/data/clean/scores/support_scores[_sampleX]_mXX.parquet."
        ),
    )
    p.add_argument(
        "--write-config-json",
        action="store_true",
        help="Write selected_support_score_config.json next to the exported parquet.",
    )
    return p.parse_args()


def build_frame_state_path(frame_state_dir: str, frame_state_name: str, sample_frac: Optional[float]) -> str:
    return os.path.join(frame_state_dir, f"{frame_state_name}{format_sample_suffix(sample_frac)}.parquet")


def compute_one_config(df: pd.DataFrame, *, start_minute: float, max_minute: float, far_adc_threshold: float,
                       weights: tuple[float, float, float], min_support_frames: int,
                       xp_ratio_min: float, xp_ratio_max: float, config_id: str) -> pd.DataFrame:
    work = df[(df["minute"] >= start_minute) & (df["minute"] <= max_minute)].copy()
    if work.empty:
        return pd.DataFrame(columns=JOIN_KEYS + ["config_id"])

    # update final XP before spatial filters: use all frames in window
    xp_last = (
        work.sort_values(["match_id", "team_id", "frame_idx"])
        .groupby(JOIN_KEYS, as_index=False)
        .agg(
            support_adc_xp_ratio_v2=("support_xp", "last"),
            adc_xp_last=("adc_xp", "last"),
        )
    )
    xp_last["support_adc_xp_ratio_v2"] = np.where(
        xp_last["adc_xp_last"].fillna(0) > 0,
        xp_last["support_adc_xp_ratio_v2"] / xp_last["adc_xp_last"],
        np.nan,
    )
    xp_last = xp_last.drop(columns=["adc_xp_last"])

    # spatially valid support frames
    spatial = work[
        work["support_alive"].fillna(False)
        & work["support_x"].notna()
        & ~work["support_in_base"].fillna(False)
    ].copy()

    if spatial.empty:
        return pd.DataFrame(columns=JOIN_KEYS + ["config_id"])

    spatial["support_in_bot_extended"] = spatial["support_in_bot_extended"].fillna(False)
    spatial["out_bot"] = ~spatial["support_in_bot_extended"]

    coop = spatial[
        spatial["adc_alive"].fillna(False)
        & spatial["adc_x"].notna()
        & ~spatial["adc_in_base"].fillna(False)
    ].copy()
    coop["far_from_adc"] = coop["dist_to_adc"].fillna(-1) >= far_adc_threshold

    agg_spatial = spatial.groupby(JOIN_KEYS, as_index=False).agg(
        support_champion_name=("support_champion_name", "first"),
        adc_champion_name=("adc_champion_name", "first"),
        side=("side", "first"),
        patch=("patch", "first"),
        valid_support_frames_v2=("frame_idx", "count"),
        frames_out_bot_extended=("out_bot", "sum"),
    )
    agg_spatial["outside_ratio"] = agg_spatial["frames_out_bot_extended"] / agg_spatial["valid_support_frames_v2"]

    agg_coop = coop.groupby(JOIN_KEYS, as_index=False).agg(
        valid_coop_frames_v2=("frame_idx", "count"),
        frames_far_from_adc=("far_from_adc", "sum"),
        mean_distance_to_adc_v2=("dist_to_adc", "mean"),
    )
    if not agg_coop.empty:
        agg_coop["far_ratio"] = agg_coop["frames_far_from_adc"] / agg_coop["valid_coop_frames_v2"]

    out = agg_spatial.merge(agg_coop, on=JOIN_KEYS, how="left")
    out = out.merge(xp_last, on=JOIN_KEYS, how="left")

    # min frames filter
    out = out[out["valid_support_frames_v2"] >= min_support_frames].copy()
    if out.empty:
        return out

    xp_ratio = out["support_adc_xp_ratio_v2"].copy()
    xp_ratio = xp_ratio.clip(lower=xp_ratio_min, upper=xp_ratio_max)
    out["xp_gap"] = 1.0 - ((xp_ratio - xp_ratio_min) / (xp_ratio_max - xp_ratio_min))
    out.loc[out["support_adc_xp_ratio_v2"].isna(), "xp_gap"] = np.nan

    w_outside, w_far, w_xp = weights
    components = out[["outside_ratio", "far_ratio", "xp_gap"]].astype(float)
    weights_arr = np.asarray([w_outside, w_far, w_xp], dtype=float)
    valid_mask = components.notna().to_numpy(dtype=float)
    weighted_values = components.fillna(0.0).to_numpy(dtype=float) * weights_arr
    score_den = valid_mask * weights_arr
    den = score_den.sum(axis=1)
    out["support_roam_score_v2"] = np.where(den > 0, weighted_values.sum(axis=1) / den, np.nan)
    out["support_score_confidence_v2"] = np.minimum(1.0, out["valid_support_frames_v2"] / 6.0)

    out["config_id"] = config_id
    out["start_minute"] = float(start_minute)
    out["max_minute"] = float(max_minute)
    out["far_adc_threshold"] = float(far_adc_threshold)
    out["w_outside"] = float(w_outside)
    out["w_far"] = float(w_far)
    out["w_xp"] = float(w_xp)
    out["window_tag"] = format_window_tag(max_minute)
    return out


def default_export_path(sample_frac: Optional[float], max_minute: float) -> str:
    suffix = format_sample_suffix(sample_frac)
    return os.path.join(
        "ProgresoActual",
        "data",
        "clean",
        "scores",
        f"support_scores{suffix}_{format_window_tag(max_minute)}.parquet",
    )


def select_export_config(
    summary_df: pd.DataFrame,
    export_config_id: Optional[str],
    export_best: Optional[str],
) -> Optional[str]:
    if summary_df.empty:
        return None
    non_empty = summary_df[summary_df["rows"] > 0].copy()
    if non_empty.empty:
        return None
    if export_config_id:
        if export_config_id not in set(non_empty["config_id"]):
            raise SystemExit(f"--export-config-id no existe o no tiene filas: {export_config_id}")
        return export_config_id
    if not export_best:
        return None
    if export_best == "score_iqr":
        non_empty["score_iqr"] = non_empty["score_p75"] - non_empty["score_p25"]
    sort_cols = {
        "coverage": ["coverage", "rows"],
        "score_std": ["score_std", "coverage"],
        "score_iqr": ["score_iqr", "coverage"],
    }[export_best]
    return str(non_empty.sort_values(sort_cols, ascending=[False] * len(sort_cols)).iloc[0]["config_id"])


def export_canonical_support_scores(
    long_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    selected_config_id: str,
    out_path: str,
    write_config_json: bool,
) -> None:
    selected = long_df[long_df["config_id"] == selected_config_id].copy()
    if selected.empty:
        raise SystemExit(f"No hay filas para exportar config_id={selected_config_id}")

    keep_cols = [
        "match_id", "team_id", "side", "patch",
        "support_champion_name", "adc_champion_name",
        "valid_support_frames_v2", "valid_coop_frames_v2",
        "outside_ratio", "far_ratio", "xp_gap",
        "frames_out_bot_extended", "frames_far_from_adc",
        "mean_distance_to_adc_v2", "support_adc_xp_ratio_v2",
        "support_score_confidence_v2", "support_roam_score_v2",
        "config_id", "start_minute", "max_minute", "far_adc_threshold",
        "w_outside", "w_far", "w_xp", "window_tag",
    ]
    keep_cols = [c for c in keep_cols if c in selected.columns]
    selected = selected[keep_cols].sort_values(["match_id", "team_id"]).reset_index(drop=True)
    selected["support_v2_definition"] = "weighted_mean(outside_ratio, far_ratio, xp_gap)"
    selected["support_roam_score_v2"] = selected["support_roam_score_v2"].clip(0.0, 1.0)

    out = Path(out_path)
    ensure_dir(str(out.parent))
    selected.to_parquet(out, index=False)
    print(f"[Saved] Canonical support scores: {os.path.abspath(out)}")

    if write_config_json:
        config = summary_df[summary_df["config_id"] == selected_config_id].iloc[0].to_dict()
        config["exported_rows"] = int(len(selected))
        config["exported_path"] = os.path.abspath(out)
        config_path = out.with_name("selected_support_score_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"[Saved] Selected config JSON: {os.path.abspath(config_path)}")


def main() -> None:
    args = parse_args()
    frame_state_path = build_frame_state_path(args.frame_state_dir, args.frame_state_name, args.sample_frac)
    if not os.path.exists(frame_state_path):
        raise SystemExit(f"No existe el frame-state parquet: {frame_state_path}")

    ensure_dir(args.outdir)
    print(f"[Input] {os.path.abspath(frame_state_path)}")
    print(f"[Outdir] {os.path.abspath(args.outdir)}")

    df = pd.read_parquet(frame_state_path)
    print(f"Rows frame-state: {len(df)}")

    weight_triplets = parse_weight_triplets(args.weight_triplets)
    configs = []
    for idx, (start_minute, max_minute, far_thr, weights) in enumerate(
        itertools.product(args.start_minutes, args.max_minutes, args.far_adc_thresholds, weight_triplets),
        start=1,
    ):
        if start_minute >= max_minute:
            continue
        config_id = f"cfg_{idx:03d}_s{str(start_minute).replace('.','p')}_m{int(round(max_minute)):02d}_far{int(round(far_thr))}_w{str(weights[0]).replace('.','p')}-{str(weights[1]).replace('.','p')}-{str(weights[2]).replace('.','p')}"
        configs.append((config_id, start_minute, max_minute, far_thr, weights))

    if not configs:
        raise SystemExit("No hay configuraciones válidas. Revisa start_minutes y max_minutes.")

    long_parts = []
    summary_rows = []
    champ_parts = []

    for config_id, start_minute, max_minute, far_thr, weights in configs:
        scored = compute_one_config(
            df,
            start_minute=start_minute,
            max_minute=max_minute,
            far_adc_threshold=far_thr,
            weights=weights,
            min_support_frames=args.min_support_frames,
            xp_ratio_min=args.xp_ratio_min,
            xp_ratio_max=args.xp_ratio_max,
            config_id=config_id,
        )
        if scored.empty:
            summary_rows.append({
                "config_id": config_id,
                "start_minute": start_minute,
                "max_minute": max_minute,
                "far_adc_threshold": far_thr,
                "w_outside": weights[0],
                "w_far": weights[1],
                "w_xp": weights[2],
                "rows": 0,
                "coverage": 0.0,
                "score_mean": np.nan,
                "score_std": np.nan,
                "score_p25": np.nan,
                "score_median": np.nan,
                "score_p75": np.nan,
            })
            continue

        long_parts.append(scored)
        s = pd.to_numeric(scored["support_roam_score_v2"], errors="coerce").dropna()
        summary_rows.append({
            "config_id": config_id,
            "start_minute": start_minute,
            "max_minute": max_minute,
            "far_adc_threshold": far_thr,
            "w_outside": weights[0],
            "w_far": weights[1],
            "w_xp": weights[2],
            "rows": int(len(scored)),
            "coverage": float(len(scored) / df[JOIN_KEYS].drop_duplicates().shape[0]),
            "score_mean": float(s.mean()) if not s.empty else np.nan,
            "score_std": float(s.std(ddof=0)) if not s.empty else np.nan,
            "score_p25": float(s.quantile(0.25)) if not s.empty else np.nan,
            "score_median": float(s.median()) if not s.empty else np.nan,
            "score_p75": float(s.quantile(0.75)) if not s.empty else np.nan,
            "score_iqr": float(s.quantile(0.75) - s.quantile(0.25)) if not s.empty else np.nan,
        })

        if args.champion_summary and "support_champion_name" in scored.columns:
            champ = (
                scored.groupby("support_champion_name", dropna=False)["support_roam_score_v2"]
                .agg(["count", "mean", "median", "std"]).reset_index()
            )
            champ.insert(0, "config_id", config_id)
            champ_parts.append(champ)

    long_df = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)

    suffix = format_sample_suffix(args.sample_frac)
    long_path = os.path.join(args.outdir, f"support_score_grid_long{suffix}.parquet")
    summary_path = os.path.join(args.outdir, f"support_score_grid_summary{suffix}.csv")
    long_df.to_parquet(long_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n[Saved] Long parquet: {os.path.abspath(long_path)}")
    print(f"[Saved] Summary CSV:  {os.path.abspath(summary_path)}")
    if not summary_df.empty:
        print("\nTop configs by coverage then rows:")
        print(summary_df.sort_values(["coverage", "rows"], ascending=[False, False]).head(10).to_string(index=False))

    selected_config_id = select_export_config(summary_df, args.export_config_id, args.export_best)
    if selected_config_id:
        selected_summary = summary_df[summary_df["config_id"] == selected_config_id].iloc[0]
        export_path = args.export_support_scores_path or default_export_path(args.sample_frac, float(selected_summary["max_minute"]))
        export_canonical_support_scores(
            long_df=long_df,
            summary_df=summary_df,
            selected_config_id=selected_config_id,
            out_path=export_path,
            write_config_json=args.write_config_json,
        )
    elif args.export_support_scores_path:
        raise SystemExit("--export-support-scores-path requiere --export-config-id o --export-best.")

    if champ_parts:
        champ_df = pd.concat(champ_parts, ignore_index=True)
        champ_path = os.path.join(args.outdir, f"support_score_grid_champion_summary{suffix}.csv")
        champ_df.to_csv(champ_path, index=False)
        print(f"[Saved] Champion summary: {os.path.abspath(champ_path)}")


if __name__ == "__main__":
    main()
