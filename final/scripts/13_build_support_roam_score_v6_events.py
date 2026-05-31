#!/usr/bin/env python3
"""
13_build_support_roam_score_v6_events.py

Build event-enriched support roaming labels.

The selected v6 label keeps the v5 frame-based score as the backbone and adds a
small amount of active evidence from Riot timeline events. Event columns are
postgame target-building evidence; they must not be used as model inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V5 = REPO_ROOT / "final" / "data" / "scores" / "support_scores_v5_geometry_m12.parquet"
DEFAULT_EVENTS = REPO_ROOT / "final" / "data" / "event_context" / "support_event_context_m12.parquet"
DEFAULT_OUTDIR = REPO_ROOT / "final" / "analysis" / "label_v6_events"
DEFAULT_EXPORT_DIR = REPO_ROOT / "final" / "data" / "scores"

JOIN_KEYS = ["match_id", "team_id"]
V5_RAW = "raw_support_roam_score_v5_geometry"
V5_SCORE = "support_roam_score_v5_geometry"
RAW_SCORE_COL = "raw_support_roam_score_v6_events"
SCORE_COL = "support_roam_score_v6_events"
VARIANT_WIDE_OUT = "support_scores_v6_event_variants_m12.parquet"

VARIANT_RECIPES: List[Dict[str, Any]] = [
    {
        "variant_id": "v5_frame_rebuild",
        "frame_source": "base",
        "w_frame": 1.00,
        "w_combat": 0.00,
        "w_vision": 0.00,
        "description": "v5 raw frame score rebuilt with the v6 gamma for direct calibration checks",
    },
    {
        "variant_id": "frame_no_xp",
        "frame_source": "no_xp",
        "w_frame": 1.00,
        "w_combat": 0.00,
        "w_vision": 0.00,
        "description": "frame-only ablation without the xp-gap component",
    },
    {
        "variant_id": "frame_no_far",
        "frame_source": "no_far",
        "w_frame": 1.00,
        "w_combat": 0.00,
        "w_vision": 0.00,
        "description": "frame-only ablation without the support-ADC distance component",
    },
    {
        "variant_id": "frame_no_outside",
        "frame_source": "no_outside",
        "w_frame": 1.00,
        "w_combat": 0.00,
        "w_vision": 0.00,
        "description": "frame-only ablation without the outside-bot-context component",
    },
    {
        "variant_id": "events_tiny_90_07_03",
        "frame_source": "base",
        "w_frame": 0.90,
        "w_combat": 0.07,
        "w_vision": 0.03,
        "description": "mostly v5 frame score plus very weak event evidence",
    },
    {
        "variant_id": "events_light_85_10_05",
        "frame_source": "base",
        "w_frame": 0.85,
        "w_combat": 0.10,
        "w_vision": 0.05,
        "description": "v5 frame score plus light combat and vision event evidence",
    },
    {
        "variant_id": "events_selected_75_15_10",
        "frame_source": "base",
        "w_frame": 0.75,
        "w_combat": 0.15,
        "w_vision": 0.10,
        "description": "selected v6 recipe: v5 frame score plus weak timeline event evidence",
    },
    {
        "variant_id": "events_equal_70_15_15",
        "frame_source": "base",
        "w_frame": 0.70,
        "w_combat": 0.15,
        "w_vision": 0.15,
        "description": "balanced event channels with frame score still dominant",
    },
    {
        "variant_id": "events_balanced_65_20_15",
        "frame_source": "base",
        "w_frame": 0.65,
        "w_combat": 0.20,
        "w_vision": 0.15,
        "description": "stronger event-enriched label with frame score as backbone",
    },
    {
        "variant_id": "events_heavy_50_30_20",
        "frame_source": "base",
        "w_frame": 0.50,
        "w_combat": 0.30,
        "w_vision": 0.20,
        "description": "stress-test label where event evidence has large influence",
    },
    {
        "variant_id": "events_combat_only_80_20",
        "frame_source": "base",
        "w_frame": 0.80,
        "w_combat": 0.20,
        "w_vision": 0.00,
        "description": "event ablation using combat evidence only",
    },
    {
        "variant_id": "events_vision_only_80_20",
        "frame_source": "base",
        "w_frame": 0.80,
        "w_combat": 0.00,
        "w_vision": 0.20,
        "description": "event ablation using generic support vision activity only",
    },
    {
        "variant_id": "events_combat_strong_65_35",
        "frame_source": "base",
        "w_frame": 0.65,
        "w_combat": 0.35,
        "w_vision": 0.00,
        "description": "combat-heavy event ablation",
    },
    {
        "variant_id": "events_vision_strong_70_30",
        "frame_source": "base",
        "w_frame": 0.70,
        "w_combat": 0.00,
        "w_vision": 0.30,
        "description": "vision-heavy event ablation",
    },
    {
        "variant_id": "events_only_60_40",
        "frame_source": "base",
        "w_frame": 0.00,
        "w_combat": 0.60,
        "w_vision": 0.40,
        "description": "pure event sanity check; not recommended as the main label",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build support_roam_score v6 with event evidence.")
    p.add_argument("--v5-scores", default=str(DEFAULT_V5))
    p.add_argument("--event-context", default=str(DEFAULT_EVENTS))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    p.add_argument("--selected-out-name", default="support_scores_v6_events_m12.parquet")
    p.add_argument("--gamma", type=float, default=0.75)
    p.add_argument("--w-frame", type=float, default=0.75)
    p.add_argument("--w-combat", type=float, default=0.15)
    p.add_argument("--w-vision", type=float, default=0.10)
    p.add_argument("--export-scores", action="store_true")
    return p.parse_args()


def clip01(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)


def weighted_average(parts: List[np.ndarray], weights: List[float]) -> np.ndarray:
    values = np.vstack(parts).T
    w = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values)
    num = np.nan_to_num(values, nan=0.0) @ w
    den = valid.astype(float) @ w
    return np.where(den > 0, num / den, np.nan)


def add_event_components(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "support_wards_0_12",
        "support_ward_kills_0_12",
        "support_kill_assists_out_bot_0_12",
        "support_deaths_out_bot_0_12",
        "team_objectives_out_bot_0_12",
        "botlane_deaths_bot_0_12",
    ]:
        if col not in out.columns:
            out[col] = 0
        out[col] = out[col].fillna(0)

    # Ward events in this Riot export have no position, so this is intentionally
    # weak evidence: useful as support activity, not as spatial proof of roam.
    out["vision_event_score_v6"] = clip01(
        (out["support_wards_0_12"] + 0.5 * out["support_ward_kills_0_12"]) / 8.0
    )
    out["combat_event_score_v6"] = clip01(
        (
            out["support_kill_assists_out_bot_0_12"]
            + 0.5 * out["support_deaths_out_bot_0_12"]
            + 0.5 * out["team_objectives_out_bot_0_12"]
        )
        / 2.0
    )
    out["chaos_score_v6"] = clip01(out["botlane_deaths_bot_0_12"] / 6.0)
    out["active_event_score_v6"] = weighted_average(
        [
            out["combat_event_score_v6"].to_numpy(dtype=np.float64),
            out["vision_event_score_v6"].to_numpy(dtype=np.float64),
        ],
        [0.60, 0.40],
    )
    return out


def frame_component(df: pd.DataFrame, w_outside: float, w_far: float, w_xp: float) -> np.ndarray:
    parts = [
        df["outside_ratio_v5"].to_numpy(dtype=np.float64),
        df["far_ratio_v5"].to_numpy(dtype=np.float64),
        df["xp_gap_v5"].to_numpy(dtype=np.float64),
    ]
    return weighted_average(parts, [w_outside, w_far, w_xp])


def add_variant(
    df: pd.DataFrame,
    variant_id: str,
    frame_raw: np.ndarray,
    w_frame: float,
    w_combat: float,
    w_vision: float,
    gamma: float,
) -> pd.DataFrame:
    raw = weighted_average(
        [
            frame_raw,
            df["combat_event_score_v6"].to_numpy(dtype=np.float64),
            df["vision_event_score_v6"].to_numpy(dtype=np.float64),
        ],
        [w_frame, w_combat, w_vision],
    )
    score = np.power(clip01(raw), gamma)
    return pd.DataFrame(
        {
            "match_id": df["match_id"],
            "team_id": df["team_id"],
            "variant_id": variant_id,
            "raw_score": raw,
            "score": score,
            "w_frame": w_frame,
            "w_combat": w_combat,
            "w_vision": w_vision,
            "gamma": gamma,
        }
    )


def build_variants(df: pd.DataFrame, frame_sources: Dict[str, np.ndarray], gamma: float) -> pd.DataFrame:
    rows = []
    for recipe in VARIANT_RECIPES:
        rows.append(
            add_variant(
                df,
                recipe["variant_id"],
                frame_sources[recipe["frame_source"]],
                recipe["w_frame"],
                recipe["w_combat"],
                recipe["w_vision"],
                gamma,
            )
        )
    variants = pd.concat(rows, ignore_index=True)
    description_map = {r["variant_id"]: r["description"] for r in VARIANT_RECIPES}
    frame_source_map = {r["variant_id"]: r["frame_source"] for r in VARIANT_RECIPES}
    variants["variant_description"] = variants["variant_id"].map(description_map)
    variants["frame_source"] = variants["variant_id"].map(frame_source_map)
    return variants


def save_variant_wide(
    df: pd.DataFrame,
    variants: pd.DataFrame,
    outdir: Path,
    export_dir: Path,
    export_scores: bool,
) -> None:
    wide_score = variants.pivot(index=JOIN_KEYS, columns="variant_id", values="score")
    wide_score = wide_score.rename(columns={c: f"support_roam_score_{c}" for c in wide_score.columns})
    wide_raw = variants.pivot(index=JOIN_KEYS, columns="variant_id", values="raw_score")
    wide_raw = wide_raw.rename(columns={c: f"raw_support_roam_score_{c}" for c in wide_raw.columns})
    wide = pd.concat([wide_score, wide_raw], axis=1).reset_index()

    context_cols = [
        "side", "patch", "support_champion_name", "adc_champion_name",
        "valid_support_frames_v5", "valid_coop_frames_v5",
        "outside_ratio_v5", "far_ratio_v5", "xp_gap_v5",
        "vision_event_score_v6", "combat_event_score_v6", "active_event_score_v6", "chaos_score_v6",
        "support_wards_0_12", "support_ward_kills_0_12",
        "support_kill_assists_0_12", "support_kill_assists_out_bot_0_12",
        "support_deaths_0_12", "support_deaths_out_bot_0_12",
        "adc_deaths_0_12", "botlane_deaths_bot_0_12",
        "team_objectives_0_12", "team_objectives_out_bot_0_12",
        "support_score_confidence_v5", "support_score_confidence_v6",
        V5_RAW, V5_SCORE, RAW_SCORE_COL, SCORE_COL,
    ]
    base = df[[c for c in JOIN_KEYS + context_cols if c in df.columns]].drop_duplicates(JOIN_KEYS)
    wide = base.merge(wide, on=JOIN_KEYS, how="left")
    wide_path = outdir / VARIANT_WIDE_OUT
    wide.sort_values(JOIN_KEYS).to_parquet(wide_path, index=False)
    if export_scores:
        export_dir.mkdir(parents=True, exist_ok=True)
        export_path = export_dir / VARIANT_WIDE_OUT
        wide.sort_values(JOIN_KEYS).to_parquet(export_path, index=False)
        print(f"[Exported] {export_path.resolve()}")
    print(f"[Saved] {wide_path.resolve()}")


def numeric_summary(df: pd.DataFrame, score_col: str) -> Dict[str, Any]:
    s = df[score_col].dropna()
    return {
        "n": int(s.shape[0]),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "q05": float(s.quantile(0.05)),
        "q25": float(s.quantile(0.25)),
        "median": float(s.quantile(0.50)),
        "q75": float(s.quantile(0.75)),
        "q95": float(s.quantile(0.95)),
        "max": float(s.max()),
        "share_eq_0": float((s == 0).mean()),
        "share_eq_1": float((s == 1).mean()),
    }


def save_plots(scores: pd.DataFrame, variants: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(scores[V5_SCORE].dropna(), bins=50, range=(0, 1), alpha=0.45, label="v5 geometry", color="#999999")
    ax.hist(scores[SCORE_COL].dropna(), bins=50, range=(0, 1), alpha=0.55, label="v6 events selected", color="#2f80ed")
    ax.set_title("Support roam score: v5 vs v6 events")
    ax.set_xlabel("score")
    ax.set_ylabel("match-team rows")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "support_roam_score_v5_vs_v6_events_overlay.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(scores[V5_SCORE], scores[SCORE_COL], s=2, alpha=0.08, color="#2f80ed")
    ax.plot([0, 1], [0, 1], color="black", linewidth=1)
    ax.set_title("Row-level score relation: v5 vs v6 events")
    ax.set_xlabel("v5 score")
    ax.set_ylabel("v6 events score")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "support_roam_score_v5_vs_v6_events_scatter.png", dpi=180)
    plt.close(fig)

    summary = variants.groupby("variant_id")["score"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(summary.index, summary.values, color="#2f80ed", alpha=0.85)
    ax.set_xlabel("mean score")
    ax.set_title("V6 candidate label variants")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "support_roam_score_v6_variant_means.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    export_dir = Path(args.export_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_parquet(args.v5_scores)
    events = pd.read_parquet(args.event_context)
    df = scores.merge(events, on=JOIN_KEYS, how="left", suffixes=("", "_event"))
    df = add_event_components(df)

    frame_base = df[V5_RAW].to_numpy(dtype=np.float64)
    frame_no_xp = frame_component(df, 0.45, 0.35, 0.0)
    frame_no_far = frame_component(df, 0.45, 0.0, 0.20)
    frame_no_outside = frame_component(df, 0.0, 0.35, 0.20)

    selected_raw = weighted_average(
        [
            frame_base,
            df["combat_event_score_v6"].to_numpy(dtype=np.float64),
            df["vision_event_score_v6"].to_numpy(dtype=np.float64),
        ],
        [args.w_frame, args.w_combat, args.w_vision],
    )
    df[RAW_SCORE_COL] = selected_raw
    df[SCORE_COL] = np.power(clip01(selected_raw), args.gamma)
    df["support_score_confidence_v6"] = np.clip(
        df["support_score_confidence_v5"].fillna(0.0) * (1.0 - 0.20 * df["chaos_score_v6"]),
        0.0,
        1.0,
    )
    df["variant_id_v6"] = "v6_events_selected"
    df["variant_description_v6"] = (
        "v5 frame score plus weak timeline event evidence: out-of-bot combat and generic support vision activity"
    )

    frame_sources = {
        "base": frame_base,
        "no_xp": frame_no_xp,
        "no_far": frame_no_far,
        "no_outside": frame_no_outside,
    }
    variants = build_variants(df, frame_sources, args.gamma)

    selected_summary = numeric_summary(df, SCORE_COL)
    v5_summary = numeric_summary(df, V5_SCORE)
    variant_summary = (
        variants.groupby("variant_id")
        .agg(
            rows=("score", "count"),
            mean=("score", "mean"),
            std=("score", "std"),
            q05=("score", lambda s: s.quantile(0.05)),
            median=("score", "median"),
            q95=("score", lambda s: s.quantile(0.95)),
            w_frame=("w_frame", "first"),
            w_combat=("w_combat", "first"),
            w_vision=("w_vision", "first"),
            gamma=("gamma", "first"),
            frame_source=("frame_source", "first"),
            description=("variant_description", "first"),
        )
        .reset_index()
    )
    variant_summary.to_csv(outdir / "support_roam_score_v6_variant_summary.csv", index=False)
    variants.to_parquet(outdir / "support_roam_score_v6_variants_long.parquet", index=False)

    save_plots(df, variants, outdir)

    metadata = {
        "score_col": SCORE_COL,
        "raw_score_col": RAW_SCORE_COL,
        "source_v5_scores": str(Path(args.v5_scores).resolve()),
        "source_event_context": str(Path(args.event_context).resolve()),
        "selected_recipe": {
            "w_frame": args.w_frame,
            "w_combat": args.w_combat,
            "w_vision": args.w_vision,
            "gamma": args.gamma,
            "note": "Ward events have no map position in this Riot timeline export; vision evidence is generic support activity.",
        },
        "v5_summary": v5_summary,
        "v6_summary": selected_summary,
        "row_corr_v5_v6": float(df[V5_SCORE].corr(df[SCORE_COL])),
        "mean_delta_v6_minus_v5": float((df[SCORE_COL] - df[V5_SCORE]).mean()),
        "variant_count": int(variants["variant_id"].nunique()),
        "variant_wide_out": VARIANT_WIDE_OUT,
    }
    (outdir / "support_roam_score_v6_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.export_scores:
        export_dir.mkdir(parents=True, exist_ok=True)
        keep_cols = [
            "match_id", "team_id", "side", "patch",
            "support_champion_name", "adc_champion_name",
            "valid_support_frames_v5", "valid_coop_frames_v5",
            "outside_ratio_v5", "far_ratio_v5", "xp_gap_v5",
            "vision_event_score_v6", "combat_event_score_v6", "active_event_score_v6", "chaos_score_v6",
            "support_wards_0_12", "support_ward_kills_0_12",
            "support_kill_assists_0_12", "support_kill_assists_out_bot_0_12",
            "support_deaths_0_12", "support_deaths_out_bot_0_12",
            "adc_deaths_0_12", "botlane_deaths_bot_0_12",
            "team_objectives_0_12", "team_objectives_out_bot_0_12",
            "support_score_confidence_v5", "support_score_confidence_v6",
            V5_RAW, V5_SCORE, RAW_SCORE_COL, SCORE_COL,
            "variant_id_v6", "variant_description_v6",
        ]
        export_path = export_dir / args.selected_out_name
        df[[c for c in keep_cols if c in df.columns]].sort_values(JOIN_KEYS).to_parquet(export_path, index=False)
        print(f"[Exported] {export_path.resolve()}")

    save_variant_wide(df, variants, outdir, export_dir, args.export_scores)

    print(f"[Saved] {outdir.resolve()}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
