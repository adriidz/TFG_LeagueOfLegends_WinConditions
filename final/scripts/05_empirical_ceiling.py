#!/usr/bin/env python3
"""
05_empirical_ceiling.py -- Estimate the predictive ceiling from draft composition.

Quantifies how much of the roaming score variance is explained by composition
vs match-level noise. Uses ANOVA-style variance decomposition and ICC.

Three granularity levels:
  1. Exact champion IDs
  2. Riot official classes (6 categories)
  3. Community archetypes (engage_tank, enchanter, ganker, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = str(REPO_ROOT / "final" / "data" / "training" / "train.parquet")
DEFAULT_OUTDIR = str(REPO_ROOT / "final" / "analysis" / "ceiling")
DEFAULT_CLASSES = str(REPO_ROOT / "final" / "data" / "champion_classes.json")
DEFAULT_ARCHETYPES = str(REPO_ROOT / "final" / "data" / "champion_archetypes.json")

TARGET_COL = "support_roam_score"
ROLE_KEYS = ("top", "jungle", "middle", "bottom", "utility")
SIDES = ("ally", "enemy")

# Map our column role names to archetype JSON keys
ROLE_TO_ARCH_KEY = {
    "top": "top", "jungle": "jungle", "middle": "mid",
    "bottom": "bottom", "utility": "support",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Empirical ceiling analysis.")
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--champion-classes", default=DEFAULT_CLASSES)
    p.add_argument("--champion-archetypes", default=DEFAULT_ARCHETYPES)
    p.add_argument("--min-group-size", type=int, default=5)
    return p.parse_args()


def icc_oneway(groups: pd.Series, values: pd.Series, min_size: int) -> Dict[str, Any]:
    """ICC(1) = (MSB - MSW) / (MSB + (k-1)*MSW)."""
    df = pd.DataFrame({"group": groups, "value": values}).dropna()
    counts = df.groupby("group")["value"].count()
    valid_groups = counts[counts >= min_size].index
    df = df[df["group"].isin(valid_groups)]

    if len(df) < 10 or df["group"].nunique() < 3:
        return {"icc": float("nan"), "n_groups": 0, "n_rows": 0,
                "mean_group_size": float("nan"), "note": "insufficient data"}

    n = len(df)
    k_groups = df["group"].nunique()
    grand_mean = df["value"].mean()
    group_means = df.groupby("group")["value"].mean()
    group_sizes = df.groupby("group")["value"].count()

    ssb = float(((group_means - grand_mean) ** 2 * group_sizes).sum())
    ssw = float(df.groupby("group")["value"].apply(
        lambda x: ((x - x.mean()) ** 2).sum()
    ).sum())

    df_between = k_groups - 1
    df_within = n - k_groups
    msb = ssb / df_between if df_between > 0 else 0
    msw = ssw / df_within if df_within > 0 else 0
    k_avg = float(group_sizes.mean())
    icc_val = (msb - msw) / (msb + (k_avg - 1) * msw) if (msb + (k_avg - 1) * msw) > 0 else 0

    return {
        "icc": float(np.clip(icc_val, 0.0, 1.0)),
        "n_groups": int(k_groups),
        "n_rows": int(n),
        "mean_group_size": float(k_avg),
        "var_between": float(msb),
        "var_within": float(msw),
        "total_var": float(df["value"].var()),
    }


def grouping_key(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    parts = [df[c].astype(str) for c in columns if c in df.columns]
    return parts[0] if len(parts) == 1 else pd.Series(
        ["_".join(vals) for vals in zip(*parts)], index=df.index
    )


def add_class_columns(df: pd.DataFrame, class_map: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for s in SIDES:
        for r in ROLE_KEYS:
            id_col = f"{s}_{r}_champion_id"
            cls_col = f"{s}_{r}_class"
            if id_col in out.columns:
                out[cls_col] = out[id_col].astype(str).map(class_map).fillna("Unknown")
    return out


def add_archetype_columns(
    df: pd.DataFrame, arch_champs: Dict[str, dict], class_map: Dict[str, str]
) -> pd.DataFrame:
    """Add _archetype columns using community archetypes (role-aware)."""
    out = df.copy()
    for s in SIDES:
        for r in ROLE_KEYS:
            id_col = f"{s}_{r}_champion_id"
            arch_col = f"{s}_{r}_archetype"
            if id_col not in out.columns:
                continue
            role_key = ROLE_TO_ARCH_KEY[r]

            def lookup(cid, _rk=role_key):
                cid_str = str(int(cid)) if not pd.isna(cid) else ""
                entry = arch_champs.get(cid_str, {})
                if _rk in entry:
                    return entry[_rk]
                if "generic" in entry:
                    return entry["generic"]
                if cid_str in class_map:
                    return class_map[cid_str].lower()
                return "other"

            out[arch_col] = out[id_col].apply(lookup)
    return out


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.train)
    print(f"[Data] rows={len(df):,}  target_std={df[TARGET_COL].std():.4f}")

    # --- Load Riot classes ---
    class_map = {}
    classes_path = Path(args.champion_classes)
    if classes_path.exists():
        raw = json.loads(classes_path.read_text(encoding="utf-8"))
        class_map = {k: v["primary_class"] for k, v in raw.items()}
        df = add_class_columns(df, class_map)
        print(f"[Riot Classes] {len(class_map)} champions")

    # --- Load community archetypes ---
    has_archetypes = False
    arch_path = Path(args.champion_archetypes)
    if arch_path.exists():
        arch_raw = json.loads(arch_path.read_text(encoding="utf-8"))
        arch_champs = arch_raw.get("champions", {})
        df = add_archetype_columns(df, arch_champs, class_map)
        has_archetypes = True
        print(f"[Archetypes] {len(arch_champs)} champions mapped")
        if "ally_utility_archetype" in df.columns:
            dist = df["ally_utility_archetype"].value_counts().to_dict()
            print(f"  Support archetypes: {dist}")

    # =============================================
    #  GROUPINGS
    # =============================================
    groupings = {}

    # --- Exact champion IDs ---
    groupings["support_champion"] = ["ally_utility_champion_id"]
    groupings["support_champion+side"] = ["ally_utility_champion_id", "side"]
    groupings["botlane_champions"] = ["ally_utility_champion_id", "ally_bottom_champion_id"]
    groupings["botlane_champions+side"] = ["ally_utility_champion_id", "ally_bottom_champion_id", "side"]
    groupings["sup_vs_enemy_sup_champion"] = ["ally_utility_champion_id", "enemy_utility_champion_id"]

    # --- Riot classes (6 categories) ---
    if class_map:
        groupings["support_riot_class"] = ["ally_utility_class"]
        groupings["botlane_riot_classes"] = ["ally_utility_class", "ally_bottom_class"]
        groupings["all_10_riot_classes"] = [f"{s}_{r}_class" for s in SIDES for r in ROLE_KEYS]

    # --- Community archetypes ---
    if has_archetypes:
        groupings["support_archetype"] = ["ally_utility_archetype"]
        groupings["support_archetype+side"] = ["ally_utility_archetype", "side"]
        groupings["botlane_archetypes"] = ["ally_utility_archetype", "ally_bottom_archetype"]
        groupings["botlane_archetypes+side"] = ["ally_utility_archetype", "ally_bottom_archetype", "side"]
        groupings["sup_vs_enemy_sup_archetype"] = ["ally_utility_archetype", "enemy_utility_archetype"]
        groupings["sup+jungle_archetypes"] = ["ally_utility_archetype", "ally_jungle_archetype"]
        groupings["sup+jungle_archetypes+side"] = ["ally_utility_archetype", "ally_jungle_archetype", "side"]
        groupings["sup+jungle+top_archetypes"] = [
            "ally_utility_archetype", "ally_jungle_archetype", "ally_top_archetype",
        ]
        groupings["botlane_vs_enemy_bot_archetypes"] = [
            "ally_utility_archetype", "ally_bottom_archetype",
            "enemy_utility_archetype", "enemy_bottom_archetype",
        ]
        groupings["ally_team_archetypes"] = [f"ally_{r}_archetype" for r in ROLE_KEYS]
        groupings["ally_team_archetypes+side"] = [f"ally_{r}_archetype" for r in ROLE_KEYS] + ["side"]
        groupings["all_10_archetypes"] = [f"{s}_{r}_archetype" for s in SIDES for r in ROLE_KEYS]
        groupings["all_10_archetypes+side"] = [f"{s}_{r}_archetype" for s in SIDES for r in ROLE_KEYS] + ["side"]

    # =============================================
    #  COMPUTE
    # =============================================
    results = []
    for name, cols in groupings.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"[Skip] {name}: missing {missing}")
            continue

        key = grouping_key(df, cols)
        icc_result = icc_oneway(key, df[TARGET_COL], min_size=args.min_group_size)
        icc_result["grouping"] = name
        icc_result["columns"] = cols

        # R2 of predicting group mean
        group_means = df.groupby(key)[TARGET_COL].transform("mean")
        ss_res = float(((df[TARGET_COL] - group_means) ** 2).sum())
        ss_tot = float(((df[TARGET_COL] - df[TARGET_COL].mean()) ** 2).sum())
        icc_result["r2_group_mean"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0

        results.append(icc_result)
        print(f"[{name:45s}] ICC={icc_result['icc']:.4f}  "
              f"R2={icc_result['r2_group_mean']:.4f}  "
              f"groups={icc_result['n_groups']:>6}  "
              f"rows={icc_result['n_rows']:>8,}")

    # Save
    (outdir / "ceiling_analysis.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary_df = pd.DataFrame([
        {
            "grouping": r["grouping"],
            "icc": r["icc"],
            "r2_group_mean": r.get("r2_group_mean", float("nan")),
            "n_groups": r["n_groups"],
            "n_rows": r["n_rows"],
            "mean_group_size": r["mean_group_size"],
        }
        for r in results
    ])
    summary_df.to_csv(outdir / "ceiling_summary.csv", index=False)

    # Markdown
    header = "| " + " | ".join(summary_df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(summary_df.columns)) + " |"
    rows_md = []
    for _, row in summary_df.iterrows():
        cells = []
        for c in summary_df.columns:
            v = row[c]
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        rows_md.append("| " + " | ".join(cells) + " |")
    md_table = "\n".join([header, sep] + rows_md)
    (outdir / "ceiling_summary.md").write_text(
        f"# Empirical Ceiling Analysis\n\n{md_table}\n", encoding="utf-8"
    )

    print(f"\n[Saved] {outdir.resolve()}")

    # Interpretation
    supp_champ = next((r for r in results if r["grouping"] == "support_champion"), None)
    supp_arch = next((r for r in results if r["grouping"] == "support_archetype"), None)
    supp_class = next((r for r in results if r["grouping"] == "support_riot_class"), None)
    print("\n  === COMPARISON: Support grouping granularity ===")
    if supp_class:
        print(f"  Riot class (6 categories):      ICC={supp_class['icc']:.4f}  groups={supp_class['n_groups']}")
    if supp_arch:
        print(f"  Community archetype (~7 types):  ICC={supp_arch['icc']:.4f}  groups={supp_arch['n_groups']}")
    if supp_champ:
        print(f"  Exact champion ID (~144 champs): ICC={supp_champ['icc']:.4f}  groups={supp_champ['n_groups']}")
    print("  (Higher ICC = more variance explained by group membership)")


if __name__ == "__main__":
    main()
