#!/usr/bin/env python3
"""
05_empirical_ceiling.py -- Estimate repeatable draft signal from composition.

Quantifies how much of the roaming score variance is explained by composition
vs match-level noise. ICC is kept as an in-sample descriptive consistency
metric on train. Group-mean R2 is computed out-of-sample: train group means are
applied to test rows, with the train global mean as fallback for unseen groups.

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
DEFAULT_TEST = str(REPO_ROOT / "final" / "data" / "training" / "test.parquet")
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
    p.add_argument("--test", default=DEFAULT_TEST)
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


def r2_score_manual(y_true: Any, y_pred: Any) -> float:
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true_arr - y_pred_arr) ** 2))
    ss_tot = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def group_mean_oos_r2(
    train_groups: pd.Series,
    train_values: pd.Series,
    test_groups: pd.Series,
    test_values: pd.Series,
) -> Dict[str, Any]:
    """Predict test rows with train-only group means and train global fallback."""
    train_frame = pd.DataFrame({"group": train_groups, "value": train_values}).dropna()
    test_frame = pd.DataFrame({"group": test_groups, "value": test_values}).dropna()
    if train_frame.empty or test_frame.empty:
        return {
            "r2_group_mean_oos": float("nan"),
            "n_train_groups": 0,
            "n_test_groups": 0,
            "n_test_rows": int(len(test_frame)),
            "n_unseen_test_groups": 0,
            "n_unseen_test_rows": 0,
            "train_global_mean": float("nan"),
        }

    train_global_mean = float(train_frame["value"].mean())
    train_group_means = train_frame.groupby("group")["value"].mean()
    pred = test_frame["group"].map(train_group_means)
    unseen_mask = pred.isna()
    pred = pred.fillna(train_global_mean).to_numpy(dtype=np.float64)

    return {
        "r2_group_mean_oos": r2_score_manual(test_frame["value"], pred),
        "n_train_groups": int(train_group_means.shape[0]),
        "n_test_groups": int(test_frame["group"].nunique()),
        "n_test_rows": int(len(test_frame)),
        "n_unseen_test_groups": int(test_frame.loc[unseen_mask, "group"].nunique()),
        "n_unseen_test_rows": int(unseen_mask.sum()),
        "unseen_test_row_rate": float(unseen_mask.mean()),
        "train_global_mean": train_global_mean,
    }


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

    df_train = pd.read_parquet(args.train)
    df_test = pd.read_parquet(args.test)
    print(
        f"[Data] train={len(df_train):,}  test={len(df_test):,}  "
        f"train_target_std={df_train[TARGET_COL].std():.4f}  "
        f"test_target_std={df_test[TARGET_COL].std():.4f}"
    )

    # --- Load Riot classes ---
    class_map = {}
    classes_path = Path(args.champion_classes)
    if classes_path.exists():
        raw = json.loads(classes_path.read_text(encoding="utf-8"))
        class_map = {k: v["primary_class"] for k, v in raw.items()}
        df_train = add_class_columns(df_train, class_map)
        df_test = add_class_columns(df_test, class_map)
        print(f"[Riot Classes] {len(class_map)} champions")

    # --- Load community archetypes ---
    has_archetypes = False
    arch_path = Path(args.champion_archetypes)
    if arch_path.exists():
        arch_raw = json.loads(arch_path.read_text(encoding="utf-8"))
        arch_champs = arch_raw.get("champions", {})
        df_train = add_archetype_columns(df_train, arch_champs, class_map)
        df_test = add_archetype_columns(df_test, arch_champs, class_map)
        has_archetypes = True
        print(f"[Archetypes] {len(arch_champs)} champions mapped")
        if "ally_utility_archetype" in df_train.columns:
            dist = df_train["ally_utility_archetype"].value_counts().to_dict()
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
    train_icc_results = []
    oos_results = []
    combined_results = []
    for name, cols in groupings.items():
        missing = [c for c in cols if c not in df_train.columns or c not in df_test.columns]
        if missing:
            print(f"[Skip] {name}: missing {missing}")
            continue

        train_key = grouping_key(df_train, cols)
        test_key = grouping_key(df_test, cols)
        icc_result = icc_oneway(train_key, df_train[TARGET_COL], min_size=args.min_group_size)
        icc_result["grouping"] = name
        icc_result["columns"] = cols
        icc_result["split"] = "train"
        icc_result["metric_role"] = "descriptive_in_sample_consistency"

        oos_result = group_mean_oos_r2(
            train_key,
            df_train[TARGET_COL],
            test_key,
            df_test[TARGET_COL],
        )
        oos_result["grouping"] = name
        oos_result["columns"] = cols
        oos_result["train_split"] = str(Path(args.train).resolve())
        oos_result["test_split"] = str(Path(args.test).resolve())
        oos_result["metric_role"] = "out_of_sample_group_mean_reference"
        oos_result["group_means_fit_split"] = "train"
        oos_result["predicted_split"] = "test"

        train_icc_results.append(icc_result)
        oos_results.append(oos_result)
        combined_results.append(
            {
                "grouping": name,
                "columns": cols,
                "icc_train": icc_result["icc"],
                "r2_group_mean_oos": oos_result["r2_group_mean_oos"],
                "n_train_groups_icc_min_size": icc_result["n_groups"],
                "n_train_groups_oos_means": oos_result["n_train_groups"],
                "n_test_groups": oos_result["n_test_groups"],
                "n_test_rows": oos_result["n_test_rows"],
                "n_unseen_test_groups": oos_result["n_unseen_test_groups"],
                "n_unseen_test_rows": oos_result["n_unseen_test_rows"],
            }
        )
        print(f"[{name:45s}] ICC={icc_result['icc']:.4f}  "
              f"R2_OOS={oos_result['r2_group_mean_oos']:.4f}  "
              f"train_groups={oos_result['n_train_groups']:>6}  "
              f"unseen_test_rows={oos_result['n_unseen_test_rows']:>6}")

    # Save
    (outdir / "ceiling_analysis.json").write_text(
        json.dumps(combined_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (outdir / "ceiling_train_icc.json").write_text(
        json.dumps(train_icc_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (outdir / "ceiling_oos_group_mean.json").write_text(
        json.dumps(oos_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    train_icc_df = pd.DataFrame([
        {
            "grouping": r["grouping"],
            "icc": r["icc"],
            "n_groups": r["n_groups"],
            "n_rows": r["n_rows"],
            "mean_group_size": r["mean_group_size"],
            "metric_role": r["metric_role"],
        }
        for r in train_icc_results
    ])
    oos_df = pd.DataFrame([
        {
            "grouping": r["grouping"],
            "r2_group_mean_oos": r["r2_group_mean_oos"],
            "n_train_groups": r["n_train_groups"],
            "n_test_groups": r["n_test_groups"],
            "n_test_rows": r["n_test_rows"],
            "n_unseen_test_groups": r["n_unseen_test_groups"],
            "n_unseen_test_rows": r["n_unseen_test_rows"],
            "unseen_test_row_rate": r["unseen_test_row_rate"],
            "train_global_mean": r["train_global_mean"],
            "group_means_fit_split": r["group_means_fit_split"],
            "predicted_split": r["predicted_split"],
        }
        for r in oos_results
    ])
    combined_df = pd.DataFrame(combined_results)

    train_icc_df.to_csv(outdir / "ceiling_summary_train_icc.csv", index=False)
    oos_df.to_csv(outdir / "ceiling_oos_summary.csv", index=False)

    # Backward-compatible combined summary for plotting/report scripts.
    legacy_df = combined_df.rename(
        columns={
            "icc_train": "icc",
            "r2_group_mean_oos": "r2_group_mean",
        }
    )
    legacy_df.to_csv(outdir / "ceiling_summary.csv", index=False)

    # Markdown
    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        header = "| " + " | ".join(df.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        rows_md = []
        for _, row in df.iterrows():
            cells = []
            for c in df.columns:
                v = row[c]
                cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            rows_md.append("| " + " | ".join(cells) + " |")
        return "\n".join([header, sep] + rows_md)

    train_icc_md = markdown_table(train_icc_df)
    oos_md = markdown_table(oos_df)
    combined_md = markdown_table(combined_df)
    (outdir / "ceiling_summary.md").write_text(
        "# Empirical Ceiling / Repeatable Draft Signal\n\n"
        "## Methodological note\n\n"
        "ICC and R2 are not the same metric. ICC is reported here as a descriptive "
        "in-sample train statistic: it summarizes consistency within repeated draft "
        "groups after filtering small groups. The group-mean R2 below is the model-like "
        "reference: group means are fitted only on train, applied to test, and unseen "
        "test groups fall back to the train global mean. This OOS R2 is the value that "
        "can be compared with model test R2.\n\n"
        "## Train ICC\n\n"
        f"{train_icc_md}\n\n"
        "## Out-of-Sample Group-Mean R2\n\n"
        f"{oos_md}\n\n"
        "## Combined View\n\n"
        f"{combined_md}\n",
        encoding="utf-8",
    )

    # Keep old filename with a clearer extra note.
    (outdir / "ceiling_methodology_note.md").write_text(
        "# ICC vs Out-of-Sample R2\n\n"
        "- ICC: descriptive train-only consistency metric, not a test-set model score.\n"
        "- R2 group mean OOS: train-only group means evaluated on test, with train global mean fallback for unseen groups.\n"
        "- Compare model test R2 against `ceiling_oos_summary.csv`, not against ICC directly.\n",
        encoding="utf-8",
    )

    print(f"\n[Saved] {outdir.resolve()}")

    # Interpretation
    supp_champ = next((r for r in train_icc_results if r["grouping"] == "support_champion"), None)
    supp_arch = next((r for r in train_icc_results if r["grouping"] == "support_archetype"), None)
    supp_class = next((r for r in train_icc_results if r["grouping"] == "support_riot_class"), None)
    botlane_side_oos = next(
        (r for r in oos_results if r["grouping"] == "botlane_champions+side"), None
    )
    print("\n  === COMPARISON: Support grouping granularity (train ICC) ===")
    if supp_class:
        print(f"  Riot class (6 categories):      ICC={supp_class['icc']:.4f}  groups={supp_class['n_groups']}")
    if supp_arch:
        print(f"  Community archetype (~7 types):  ICC={supp_arch['icc']:.4f}  groups={supp_arch['n_groups']}")
    if supp_champ:
        print(f"  Exact champion ID (~144 champs): ICC={supp_champ['icc']:.4f}  groups={supp_champ['n_groups']}")
    if botlane_side_oos:
        print(
            "  OOS group-mean reference "
            f"(botlane_champions+side): R2={botlane_side_oos['r2_group_mean_oos']:.4f} "
            f"unseen_test_rows={botlane_side_oos['n_unseen_test_rows']}"
        )
    print("  (ICC describes train consistency; OOS R2 is comparable with model test R2.)")


if __name__ == "__main__":
    main()
