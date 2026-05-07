#!/usr/bin/env python3
"""
Compare CPU-only variants of the support roaming label.

This script reads the frozen full support frame-state generated in
ProgresoActual, samples match_ids reproducibly for smoke runs, scores several
label variants, evaluates them without training a model, and optionally exports
the selected candidate as support_scores_v3_m12.parquet.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


JOIN_KEYS = ["match_id", "team_id"]
DEFAULT_FRAME_STATE = os.path.join(
    "ProgresoActual", "data", "clean", "frame_state", "support_frame_state.parquet"
)
DEFAULT_REFERENCE = os.path.join("ProgresoActual", "references", "champion_support_reference.csv")
DEFAULT_OUT_ROOT = os.path.join("ProgresoActual2", "analysis", "support_label_variants")
DEFAULT_EXPORT_DIR = os.path.join("ProgresoActual2", "data", "clean", "scores")
SCORE_COL = "support_roam_score_v3"
BASELINE_SCORE_COL = "support_roam_score_v2_equivalent"


@dataclass(frozen=True)
class Variant:
    variant_id: str
    family: str
    description: str
    start_minute: float = 5.0
    max_minute: float = 12.0
    far_adc_threshold: float = 2500.0
    w_outside: float = 0.45
    w_far: float = 0.35
    w_xp: float = 0.20
    transform: str = "none"
    gamma: Optional[float] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare support roam label variants without model training.")
    p.add_argument("--frame-state-path", default=DEFAULT_FRAME_STATE)
    p.add_argument("--reference-path", default=DEFAULT_REFERENCE)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--export-dir", default=DEFAULT_EXPORT_DIR)
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--sample-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-support-frames", type=int, default=2)
    p.add_argument("--xp-ratio-min", type=float, default=0.60)
    p.add_argument("--xp-ratio-max", type=float, default=1.00)
    p.add_argument("--min-champion-count", type=int, default=None)
    p.add_argument("--export-selected", action="store_true")
    p.add_argument("--selected-out-name", default="support_scores_v3_m12.parquet")
    return p.parse_args()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_name(value: object) -> str:
    return str(value).strip().lower().replace(" ", "").replace("'", "").replace(".", "")


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
        "axes.linewidth": 1.1,
    })


def load_frame_state(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Missing frame-state parquet: {path}")
    df = pd.read_parquet(p)
    required = {
        "match_id", "team_id", "side", "patch", "frame_idx", "minute",
        "support_champion_name", "adc_champion_name", "support_alive", "adc_alive",
        "support_x", "adc_x", "support_in_base", "adc_in_base",
        "support_in_bot_extended", "dist_to_adc", "support_xp", "adc_xp",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Frame-state missing required columns: {missing}")
    return df


def sample_by_match_id(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    if frac <= 0.0 or frac >= 1.0:
        return df
    match_ids = pd.Series(df["match_id"].dropna().unique())
    sampled = match_ids.sample(n=max(1, int(round(len(match_ids) * frac))), random_state=seed)
    return df[df["match_id"].isin(set(sampled))].copy()


def build_variants() -> List[Variant]:
    variants: List[Variant] = [
        Variant("v2_baseline", "baseline", "Current v2 formula: s5-m12, far2500, weights 0.45/0.35/0.20"),
    ]
    for far in (1800.0, 2000.0, 2200.0):
        variants.append(Variant(
            f"v3_threshold_far{int(far)}",
            "threshold",
            f"Baseline weights with stricter far-from-ADC threshold {int(far)}",
            far_adc_threshold=far,
        ))
    for weights in (
        (0.55, 0.35, 0.10),
        (0.50, 0.40, 0.10),
        (0.60, 0.30, 0.10),
        (0.45, 0.45, 0.10),
    ):
        token = "-".join(str(x).replace(".", "p") for x in weights)
        variants.append(Variant(
            f"v3_weights_{token}",
            "weights",
            f"Alternative component weights {weights[0]:.2f}/{weights[1]:.2f}/{weights[2]:.2f}",
            w_outside=weights[0],
            w_far=weights[1],
            w_xp=weights[2],
        ))
    for start, max_minute in ((4.0, 10.0), (4.0, 12.0), (5.0, 14.0), (6.0, 12.0)):
        variants.append(Variant(
            f"v3_window_s{int(start)}_m{int(max_minute):02d}",
            "window",
            f"Alternative time window from minute {int(start)} to {int(max_minute)}",
            start_minute=start,
            max_minute=max_minute,
        ))
    variants.append(Variant(
        "v3_calibrated_gamma075",
        "calibration",
        "Baseline formula with power calibration gamma=0.75",
        transform="gamma",
        gamma=0.75,
    ))
    variants.append(Variant(
        "v3_calibrated_q05_q95",
        "calibration",
        "Baseline formula linearly calibrated between q05 and q95",
        transform="q05_q95",
    ))
    return variants


def compute_raw_score(
    df: pd.DataFrame,
    variant: Variant,
    min_support_frames: int,
    xp_ratio_min: float,
    xp_ratio_max: float,
) -> pd.DataFrame:
    work = df[(df["minute"] >= variant.start_minute) & (df["minute"] <= variant.max_minute)].copy()
    if work.empty:
        return pd.DataFrame(columns=JOIN_KEYS)

    xp_last = (
        work.sort_values(["match_id", "team_id", "frame_idx"])
        .groupby(JOIN_KEYS, as_index=False)
        .agg(
            support_adc_xp_ratio_v3=("support_xp", "last"),
            adc_xp_last=("adc_xp", "last"),
        )
    )
    xp_last["support_adc_xp_ratio_v3"] = np.where(
        xp_last["adc_xp_last"].fillna(0) > 0,
        xp_last["support_adc_xp_ratio_v3"] / xp_last["adc_xp_last"],
        np.nan,
    )
    xp_last = xp_last.drop(columns=["adc_xp_last"])

    spatial = work[
        work["support_alive"].fillna(False)
        & work["support_x"].notna()
        & ~work["support_in_base"].fillna(False)
    ].copy()
    if spatial.empty:
        return pd.DataFrame(columns=JOIN_KEYS)

    spatial["support_in_bot_extended"] = spatial["support_in_bot_extended"].fillna(False)
    spatial["out_bot"] = ~spatial["support_in_bot_extended"]
    coop = spatial[
        spatial["adc_alive"].fillna(False)
        & spatial["adc_x"].notna()
        & ~spatial["adc_in_base"].fillna(False)
    ].copy()
    coop["far_from_adc"] = coop["dist_to_adc"].fillna(-1) >= variant.far_adc_threshold

    agg_spatial = spatial.groupby(JOIN_KEYS, as_index=False).agg(
        side=("side", "first"),
        patch=("patch", "first"),
        support_champion_name=("support_champion_name", "first"),
        adc_champion_name=("adc_champion_name", "first"),
        valid_support_frames_v3=("frame_idx", "count"),
        frames_out_bot_extended=("out_bot", "sum"),
    )
    agg_spatial["outside_ratio"] = (
        agg_spatial["frames_out_bot_extended"] / agg_spatial["valid_support_frames_v3"]
    )

    agg_coop = coop.groupby(JOIN_KEYS, as_index=False).agg(
        valid_coop_frames_v3=("frame_idx", "count"),
        frames_far_from_adc=("far_from_adc", "sum"),
        mean_distance_to_adc_v3=("dist_to_adc", "mean"),
    )
    if not agg_coop.empty:
        agg_coop["far_ratio"] = agg_coop["frames_far_from_adc"] / agg_coop["valid_coop_frames_v3"]

    out = agg_spatial.merge(agg_coop, on=JOIN_KEYS, how="left").merge(xp_last, on=JOIN_KEYS, how="left")
    out = out[out["valid_support_frames_v3"] >= min_support_frames].copy()
    if out.empty:
        return out

    xp_ratio = out["support_adc_xp_ratio_v3"].clip(lower=xp_ratio_min, upper=xp_ratio_max)
    out["xp_gap"] = 1.0 - ((xp_ratio - xp_ratio_min) / (xp_ratio_max - xp_ratio_min))
    out.loc[out["support_adc_xp_ratio_v3"].isna(), "xp_gap"] = np.nan

    components = out[["outside_ratio", "far_ratio", "xp_gap"]].astype(float)
    weights = np.asarray([variant.w_outside, variant.w_far, variant.w_xp], dtype=float)
    valid_mask = components.notna().to_numpy(dtype=float)
    weighted_values = components.fillna(0.0).to_numpy(dtype=float) * weights
    den = (valid_mask * weights).sum(axis=1)
    out["raw_support_roam_score"] = np.where(den > 0, weighted_values.sum(axis=1) / den, np.nan)
    out["support_score_confidence_v3"] = np.minimum(1.0, out["valid_support_frames_v3"] / 6.0)
    return out


def apply_transform(scores: pd.Series, variant: Variant) -> Tuple[pd.Series, Dict[str, float]]:
    valid = pd.to_numeric(scores, errors="coerce")
    meta: Dict[str, float] = {}
    if variant.transform == "gamma":
        gamma = float(variant.gamma or 1.0)
        meta["gamma"] = gamma
        return valid.clip(0.0, 1.0).pow(gamma), meta
    if variant.transform == "q05_q95":
        q05 = float(valid.quantile(0.05))
        q95 = float(valid.quantile(0.95))
        meta["calibration_q05"] = q05
        meta["calibration_q95"] = q95
        if q95 <= q05:
            return valid.clip(0.0, 1.0), meta
        return ((valid - q05) / (q95 - q05)).clip(0.0, 1.0), meta
    return valid.clip(0.0, 1.0), meta


def load_reference(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"Missing champion reference: {path}")
    ref = pd.read_csv(path)
    required = {"champion_name", "expert_support_roam_score"}
    missing = sorted(required - set(ref.columns))
    if missing:
        raise SystemExit(f"Reference missing required columns: {missing}")
    ref = ref.copy()
    ref["expert_support_roam_score"] = pd.to_numeric(ref["expert_support_roam_score"], errors="coerce")
    if "expert_confidence" in ref.columns:
        ref["expert_confidence"] = pd.to_numeric(ref["expert_confidence"], errors="coerce")
    else:
        ref["expert_confidence"] = np.nan
    ref["_join_name"] = ref["champion_name"].map(normalize_name)
    return ref


def numeric_summary(scores: pd.Series) -> Dict[str, float]:
    valid = pd.to_numeric(scores, errors="coerce").dropna()
    if valid.empty:
        return {
            "score_n": 0,
            "score_missing": int(scores.shape[0]),
        }
    return {
        "score_n": int(valid.shape[0]),
        "score_missing": int(scores.shape[0] - valid.shape[0]),
        "score_mean": float(valid.mean()),
        "score_std": float(valid.std(ddof=0)),
        "score_min": float(valid.min()),
        "score_q05": float(valid.quantile(0.05)),
        "score_q25": float(valid.quantile(0.25)),
        "score_median": float(valid.median()),
        "score_q75": float(valid.quantile(0.75)),
        "score_q95": float(valid.quantile(0.95)),
        "score_q99": float(valid.quantile(0.99)),
        "score_max": float(valid.max()),
        "share_eq_0": float((valid == 0.0).mean()),
        "share_eq_1": float((valid == 1.0).mean()),
    }


def compute_champion_means(
    scored: pd.DataFrame,
    variant: Variant,
    reference: pd.DataFrame,
    min_count: int,
) -> pd.DataFrame:
    by_champ = (
        scored.dropna(subset=[SCORE_COL])
        .groupby("support_champion_name", dropna=False)[SCORE_COL]
        .agg(generated_count="count", generated_mean="mean", generated_median="median", generated_std="std")
        .reset_index()
        .rename(columns={"support_champion_name": "champion_name"})
    )
    by_champ["_join_name"] = by_champ["champion_name"].map(normalize_name)
    merged = by_champ.merge(
        reference.drop(columns=["champion_name"]),
        on="_join_name",
        how="left",
        validate="many_to_one",
    ).drop(columns=["_join_name"])
    merged.insert(0, "variant_id", variant.variant_id)
    merged.insert(1, "family", variant.family)
    return merged[merged["generated_count"] >= min_count].copy()


def corr_metrics(champ_df: pd.DataFrame) -> Dict[str, float]:
    valid = champ_df[["expert_support_roam_score", "generated_mean"]].dropna()
    if len(valid) < 2:
        return {"expert_pearson": float("nan"), "expert_spearman": float("nan"), "expert_n": int(len(valid))}
    pearson = valid["expert_support_roam_score"].corr(valid["generated_mean"], method="pearson")
    spearman = valid["expert_support_roam_score"].corr(valid["generated_mean"], method="spearman")
    return {
        "expert_pearson": float(pearson),
        "expert_spearman": float(spearman),
        "expert_n": int(len(valid)),
    }


def champion_value(champ_df: pd.DataFrame, champion_name: str) -> float:
    key = normalize_name(champion_name)
    work = champ_df[champ_df["champion_name"].map(normalize_name) == key]
    if work.empty:
        return float("nan")
    return float(work.iloc[0]["generated_mean"])


def group_separation(champ_df: pd.DataFrame) -> Dict[str, float]:
    compared = champ_df.dropna(subset=["expert_support_roam_score", "generated_mean"]).copy()
    if compared.empty:
        return {
            "high_roam_mean": float("nan"),
            "anchored_mean": float("nan"),
            "high_minus_anchored": float("nan"),
        }
    confidence = pd.to_numeric(compared.get("expert_confidence"), errors="coerce")
    confident = confidence.fillna(1.0) >= 0.65
    high = compared[confident & (compared["expert_support_roam_score"] >= 0.70)]["generated_mean"]
    low = compared[confident & (compared["expert_support_roam_score"] <= 0.30)]["generated_mean"]
    high_mean = float(high.mean()) if not high.empty else float("nan")
    low_mean = float(low.mean()) if not low.empty else float("nan")
    return {
        "high_roam_mean": high_mean,
        "anchored_mean": low_mean,
        "high_minus_anchored": high_mean - low_mean if not math.isnan(high_mean) and not math.isnan(low_mean) else float("nan"),
    }


def side_bias(scored: pd.DataFrame) -> Dict[str, float]:
    means = scored.groupby("side")[SCORE_COL].mean()
    blue = float(means.get("blue", np.nan))
    red = float(means.get("red", np.nan))
    return {
        "blue_mean": blue,
        "red_mean": red,
        "blue_minus_red": blue - red if not math.isnan(blue) and not math.isnan(red) else float("nan"),
        "abs_blue_minus_red": abs(blue - red) if not math.isnan(blue) and not math.isnan(red) else float("nan"),
    }


def baseline_row_corr(scored: pd.DataFrame, baseline_scores: Optional[pd.DataFrame]) -> float:
    if baseline_scores is None or scored.empty:
        return float("nan")
    merged = scored[JOIN_KEYS + [SCORE_COL]].merge(
        baseline_scores[JOIN_KEYS + [BASELINE_SCORE_COL]],
        on=JOIN_KEYS,
        how="inner",
    )
    if len(merged) < 2:
        return float("nan")
    return float(merged[SCORE_COL].corr(merged[BASELINE_SCORE_COL], method="pearson"))


def compute_rank_score(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    baseline = out[out["variant_id"] == "v2_baseline"].iloc[0]
    baseline_pyke = float(baseline.get("pyke_mean", np.nan))
    baseline_bard = float(baseline.get("bard_mean", np.nan))
    baseline_q95 = float(baseline.get("score_q95", np.nan))
    baseline_side = float(baseline.get("abs_blue_minus_red", np.nan))

    out["pyke_lift"] = out["pyke_mean"] - baseline_pyke
    out["bard_lift"] = out["bard_mean"] - baseline_bard
    out["q95_lift"] = out["score_q95"] - baseline_q95
    out["side_bias_extra"] = out["abs_blue_minus_red"] - baseline_side

    out["passes_selection_rules"] = (
        (out["coverage"] >= 0.995)
        & (out["expert_spearman"] >= 0.80)
        & (out["pyke_lift"] >= 0.03)
        & (out["bard_lift"] >= 0.03)
        & (out["yuumi_mean"] <= 0.15)
        & (out["score_q95"] > baseline_q95)
        & (out["share_eq_1"] < 0.02)
        & (out["abs_blue_minus_red"] <= baseline_side + 0.015)
    )

    family_priority = {
        "threshold": 4,
        "weights": 3,
        "window": 2,
        "calibration": 1,
        "baseline": 0,
    }
    out["simplicity_priority"] = out["family"].map(family_priority).fillna(0)
    sep = out["high_minus_anchored"].fillna(out["high_minus_anchored"].median())
    out["rank_score"] = (
        2.0 * out["expert_spearman"].fillna(0.0)
        + 1.2 * sep.fillna(0.0)
        + 1.0 * out["q95_lift"].fillna(0.0)
        + 0.8 * out["pyke_lift"].fillna(0.0)
        + 0.8 * out["bard_lift"].fillna(0.0)
        - 1.5 * out["share_eq_1"].fillna(1.0)
        - 1.0 * out["side_bias_extra"].clip(lower=0).fillna(0.0)
        + 0.01 * out["simplicity_priority"]
    )
    out.loc[out["variant_id"] == "v2_baseline", "rank_score"] = -np.inf
    return out


def select_candidate(summary: pd.DataFrame) -> pd.Series:
    ranked = compute_rank_score(summary)
    strict = ranked[ranked["passes_selection_rules"]].copy()
    if not strict.empty:
        strict = strict.sort_values(["simplicity_priority", "rank_score"], ascending=[False, False])
        selected = strict.iloc[0].copy()
        selected["selection_reason"] = "passes_selection_rules"
        return selected
    fallback = ranked[ranked["variant_id"] != "v2_baseline"].sort_values("rank_score", ascending=False)
    selected = fallback.iloc[0].copy()
    selected["selection_reason"] = "best_fallback_no_strict_pass"
    return selected


def save_summary_markdown(summary: pd.DataFrame, selected: pd.Series, out_path: str) -> None:
    top_cols = [
        "variant_id", "family", "coverage", "score_median", "score_q95", "share_eq_1",
        "expert_spearman", "pyke_mean", "bard_mean", "yuumi_mean",
        "high_minus_anchored", "abs_blue_minus_red", "passes_selection_rules", "rank_score",
    ]
    ranked = compute_rank_score(summary).sort_values(
        ["passes_selection_rules", "rank_score"],
        ascending=[False, False],
    )
    table = markdown_table(ranked[top_cols].head(12))
    lines = [
        "# Support Label Variant Ranking",
        "",
        f"Selected candidate: `{selected['variant_id']}`",
        "",
        f"Selection reason: `{selected['selection_reason']}`",
        "",
        "The expert reference is used only for external validation, not for label construction.",
        "",
        "## Top variants",
        "",
        table,
        "",
    ]
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        else:
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = list(work.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in work.values.tolist():
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def save_plots(summary: pd.DataFrame, champion_means: pd.DataFrame, scored_by_variant: Dict[str, pd.DataFrame], outdir: str) -> None:
    ensure_dir(outdir)

    plt.figure(figsize=(9, 6))
    for variant_id, scored in scored_by_variant.items():
        if variant_id not in {"v2_baseline", "v3_threshold_far2000", "v3_weights_0p55-0p35-0p1", "v3_calibrated_gamma075", "v3_calibrated_q05_q95"}:
            continue
        plt.hist(scored[SCORE_COL].dropna(), bins=40, range=(0, 1), alpha=0.35, label=variant_id)
    plt.xlabel("Support roam score")
    plt.ylabel("Rows")
    plt.title("Label distribution by selected variants")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "variant_distribution_overlay.png"), dpi=180)
    plt.close()

    plot_summary = summary.sort_values("expert_spearman", ascending=False).copy()
    plt.figure(figsize=(max(10, len(plot_summary) * 0.55), 6))
    plt.bar(plot_summary["variant_id"], plot_summary["expert_spearman"], color="#276fbf")
    plt.xticks(rotation=70, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Spearman vs expert reference")
    plt.title("Champion-level rank agreement by variant")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "expert_corr_by_variant.png"), dpi=180)
    plt.close()

    key_champs = ["Pyke", "Bard", "Rell", "Alistar", "Yuumi", "Milio", "Lulu", "Soraka"]
    key = champion_means[champion_means["champion_name"].isin(key_champs)].copy()
    if not key.empty:
        pivot = key.pivot_table(index="variant_id", columns="champion_name", values="generated_mean", aggfunc="first")
        pivot = pivot.reindex(summary.sort_values("rank_score", ascending=False)["variant_id"])
        ax = pivot.plot(kind="bar", figsize=(max(11, len(pivot) * 0.65), 6), width=0.82)
        ax.set_ylabel("Champion mean generated score")
        ax.set_title("Key champion means by variant")
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=70, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "key_champions_by_variant.png"), dpi=180)
        plt.close()

    side_summary = summary.sort_values("rank_score", ascending=False)
    plt.figure(figsize=(max(10, len(side_summary) * 0.55), 6))
    plt.bar(side_summary["variant_id"], side_summary["blue_minus_red"], color="#5b8c5a")
    plt.axhline(0, color="black", linewidth=0.9)
    plt.xticks(rotation=70, ha="right")
    plt.ylabel("Blue mean - red mean")
    plt.title("Side bias by variant")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "side_bias_by_variant.png"), dpi=180)
    plt.close()


def write_selected_config(selected: pd.Series, variant: Variant, out_path: str, mode: str, source_path: str) -> None:
    payload = {
        "selected_variant": selected.to_dict(),
        "variant_definition": asdict(variant),
        "mode": mode,
        "source_frame_state_path": os.path.abspath(source_path),
        "score_col": SCORE_COL,
        "expert_reference_used_for_selection_only": True,
    }
    def clean(value):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    Path(out_path).write_text(json.dumps(clean(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    configure_plot_style()
    args = parse_args()
    outdir = os.path.join(args.out_root, args.mode)
    ensure_dir(outdir)

    print(f"[Input] frame_state={os.path.abspath(args.frame_state_path)}")
    frame_state = load_frame_state(args.frame_state_path)
    full_match_count = frame_state["match_id"].nunique()
    if args.mode == "smoke":
        frame_state = sample_by_match_id(frame_state, args.sample_frac, args.seed)
        print(
            f"[Smoke] sampled_match_ids={frame_state['match_id'].nunique()} "
            f"from full_match_ids={full_match_count} seed={args.seed}"
        )
    else:
        print(f"[Full] match_ids={full_match_count}")

    total_keys = frame_state[JOIN_KEYS].drop_duplicates().shape[0]
    reference = load_reference(args.reference_path)
    min_count = args.min_champion_count
    if min_count is None:
        min_count = 20 if args.mode == "smoke" else 100

    variants = build_variants()
    summary_rows: List[Dict[str, object]] = []
    champion_parts: List[pd.DataFrame] = []
    scored_by_variant: Dict[str, pd.DataFrame] = {}
    baseline_scores: Optional[pd.DataFrame] = None

    for variant in variants:
        print(f"[Variant] {variant.variant_id}")
        scored = compute_raw_score(
            frame_state,
            variant,
            min_support_frames=args.min_support_frames,
            xp_ratio_min=args.xp_ratio_min,
            xp_ratio_max=args.xp_ratio_max,
        )
        if scored.empty:
            continue
        scored[SCORE_COL], transform_meta = apply_transform(scored["raw_support_roam_score"], variant)
        scored[SCORE_COL] = scored[SCORE_COL].clip(0.0, 1.0)
        scored["variant_id"] = variant.variant_id
        scored["variant_family"] = variant.family
        scored["variant_description"] = variant.description
        scored["start_minute"] = variant.start_minute
        scored["max_minute"] = variant.max_minute
        scored["far_adc_threshold"] = variant.far_adc_threshold
        scored["w_outside"] = variant.w_outside
        scored["w_far"] = variant.w_far
        scored["w_xp"] = variant.w_xp
        scored["transform"] = variant.transform

        if variant.variant_id == "v2_baseline":
            baseline_scores = scored[JOIN_KEYS + [SCORE_COL]].rename(columns={SCORE_COL: BASELINE_SCORE_COL})

        champ = compute_champion_means(scored, variant, reference, min_count=min_count)
        champion_parts.append(champ)

        row: Dict[str, object] = {
            "variant_id": variant.variant_id,
            "family": variant.family,
            "description": variant.description,
            "start_minute": variant.start_minute,
            "max_minute": variant.max_minute,
            "far_adc_threshold": variant.far_adc_threshold,
            "w_outside": variant.w_outside,
            "w_far": variant.w_far,
            "w_xp": variant.w_xp,
            "transform": variant.transform,
            "rows": int(len(scored)),
            "total_match_team_keys": int(total_keys),
            "coverage": float(len(scored) / max(total_keys, 1)),
        }
        row.update(transform_meta)
        row.update(numeric_summary(scored[SCORE_COL]))
        row.update(corr_metrics(champ))
        row.update(group_separation(champ))
        row.update(side_bias(scored))
        row["row_corr_vs_v2"] = baseline_row_corr(scored, baseline_scores)
        for champ_name in ["Pyke", "Bard", "Rell", "Alistar", "Yuumi", "Milio", "Lulu", "Soraka"]:
            row[f"{normalize_name(champ_name)}_mean"] = champion_value(champ, champ_name)
        summary_rows.append(row)
        scored_by_variant[variant.variant_id] = scored

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise SystemExit("No variants produced scores.")
    summary = compute_rank_score(summary)
    selected = select_candidate(summary)
    selected_variant = next(v for v in variants if v.variant_id == selected["variant_id"])

    champion_means = pd.concat(champion_parts, ignore_index=True) if champion_parts else pd.DataFrame()
    summary = summary.sort_values(["passes_selection_rules", "rank_score"], ascending=[False, False])

    summary_path = os.path.join(outdir, "label_variant_summary.csv")
    champion_path = os.path.join(outdir, "champion_means_by_variant.csv")
    ranking_path = os.path.join(outdir, "label_variant_ranking.md")
    summary.to_csv(summary_path, index=False)
    champion_means.to_csv(champion_path, index=False)
    save_summary_markdown(summary, selected, ranking_path)
    save_plots(summary, champion_means, scored_by_variant, outdir)

    print(f"[Saved] {os.path.abspath(summary_path)}")
    print(f"[Saved] {os.path.abspath(champion_path)}")
    print(f"[Selected] {selected['variant_id']} ({selected['selection_reason']})")

    if args.export_selected:
        selected_scores = scored_by_variant[str(selected["variant_id"])].copy()
        keep_cols = [
            "match_id", "team_id", "side", "patch",
            "support_champion_name", "adc_champion_name",
            "valid_support_frames_v3", "valid_coop_frames_v3",
            "outside_ratio", "far_ratio", "xp_gap",
            "frames_out_bot_extended", "frames_far_from_adc",
            "mean_distance_to_adc_v3", "support_adc_xp_ratio_v3",
            "support_score_confidence_v3", "raw_support_roam_score", SCORE_COL,
            "variant_id", "variant_family", "variant_description",
            "start_minute", "max_minute", "far_adc_threshold",
            "w_outside", "w_far", "w_xp", "transform",
        ]
        keep_cols = [c for c in keep_cols if c in selected_scores.columns]
        ensure_dir(args.export_dir)
        export_path = os.path.join(args.export_dir, args.selected_out_name)
        selected_scores[keep_cols].sort_values(JOIN_KEYS).to_parquet(export_path, index=False)
        config_path = os.path.join(args.export_dir, "selected_support_score_v3_config.json")
        write_selected_config(selected, selected_variant, config_path, args.mode, args.frame_state_path)
        print(f"[Exported] {os.path.abspath(export_path)}")
        print(f"[Exported] {os.path.abspath(config_path)}")


if __name__ == "__main__":
    main()
