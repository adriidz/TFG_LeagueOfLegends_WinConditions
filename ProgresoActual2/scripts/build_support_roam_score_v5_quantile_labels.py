#!/usr/bin/env python3
"""
Build quantile-transformed support roam labels from manual geometry v5 raw scores.

This adapts the TransformedTargetRegressor idea to the current parquet-based
pipeline: instead of wrapping a sklearn regressor, we persist transformed target
columns that any trainer can consume.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "ProgresoActual2" / "data" / "clean" / "scores" / "support_scores_v5_geometry_m12.parquet"
DEFAULT_OUTDIR = REPO_ROOT / "ProgresoActual2" / "analysis" / "support_roam_score_v5_quantile"
DEFAULT_EXPORT_DIR = REPO_ROOT / "ProgresoActual2" / "data" / "clean" / "scores"

RAW_COL = "raw_support_roam_score_v5_geometry"
GAMMA_COL = "support_roam_score_v5_geometry"
Q_COL = "support_roam_score_v5_quantile"
Q_ZERO_COL = "support_roam_score_v5_quantile_zero_preserved"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create quantile-transformed support roam labels.")
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    p.add_argument("--raw-col", default=RAW_COL)
    p.add_argument("--n-quantiles", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-name", default="support_scores_v5_quantile_m12.parquet")
    return p.parse_args()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def fit_quantile(values: pd.Series, n_quantiles: int, seed: int) -> tuple[np.ndarray, QuantileTransformer]:
    arr = values.to_numpy(dtype=np.float64).reshape(-1, 1)
    n_q = min(int(n_quantiles), int(np.isfinite(arr[:, 0]).sum()))
    transformer = QuantileTransformer(
        n_quantiles=max(1, n_q),
        output_distribution="uniform",
        random_state=seed,
        subsample=max(int(arr.shape[0]), 1),
    )
    transformed = transformer.fit_transform(arr).reshape(-1)
    return transformed.astype(np.float64), transformer


def add_quantile_columns(df: pd.DataFrame, raw_col: str, n_quantiles: int, seed: int) -> tuple[pd.DataFrame, Dict[str, QuantileTransformer]]:
    out = df.copy()
    valid = pd.to_numeric(out[raw_col], errors="coerce")
    if valid.isna().any():
        raise SystemExit(f"Raw score column has NaNs: {raw_col}")

    q_all, transformer_all = fit_quantile(valid, n_quantiles=n_quantiles, seed=seed)
    out[Q_COL] = np.clip(q_all, 0.0, 1.0)

    positive_mask = valid > 0.0
    out[Q_ZERO_COL] = 0.0
    if positive_mask.any():
        q_pos, transformer_pos = fit_quantile(valid.loc[positive_mask], n_quantiles=n_quantiles, seed=seed)
        out.loc[positive_mask, Q_ZERO_COL] = np.clip(q_pos, 0.0, 1.0)
    else:
        transformer_pos = None

    return out, {
        "all_raw": transformer_all,
        "positive_raw": transformer_pos,
    }


def summary_for(series: pd.Series, col: str) -> Dict[str, float | str | int]:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "score_col": col,
        "n": int(valid.shape[0]),
        "mean": float(valid.mean()),
        "std": float(valid.std(ddof=0)),
        "min": float(valid.min()),
        "q01": float(valid.quantile(0.01)),
        "q05": float(valid.quantile(0.05)),
        "q25": float(valid.quantile(0.25)),
        "median": float(valid.quantile(0.50)),
        "q75": float(valid.quantile(0.75)),
        "q95": float(valid.quantile(0.95)),
        "q99": float(valid.quantile(0.99)),
        "max": float(valid.max()),
        "share_eq_0": float((valid == 0.0).mean()),
        "share_eq_1": float((valid == 1.0).mean()),
    }


def save_plots(df: pd.DataFrame, outdir: Path, raw_col: str) -> None:
    plot_cols = [raw_col]
    if GAMMA_COL in df.columns:
        plot_cols.append(GAMMA_COL)
    plot_cols.extend([Q_COL, Q_ZERO_COL])

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in plot_cols:
        ax.hist(df[col].dropna(), bins=60, range=(0, 1), alpha=0.38, label=col)
    ax.set_title("Support roam score transforms")
    ax.set_xlabel("score")
    ax.set_ylabel("match-team rows")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "support_roam_score_transform_overlay.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[raw_col], df[Q_COL], s=2, alpha=0.04, label=Q_COL)
    ax.scatter(df[raw_col], df[Q_ZERO_COL], s=2, alpha=0.04, label=Q_ZERO_COL)
    ax.set_title("Raw score to quantile-transformed score")
    ax.set_xlabel(raw_col)
    ax.set_ylabel("quantile score")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "raw_to_quantile_scatter.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    export_dir = Path(args.export_dir)
    ensure_dir(outdir)
    ensure_dir(export_dir)

    df = pd.read_parquet(args.input)
    if args.raw_col not in df.columns:
        raise SystemExit(f"Missing raw score column: {args.raw_col}")

    out, transformers = add_quantile_columns(
        df,
        raw_col=args.raw_col,
        n_quantiles=args.n_quantiles,
        seed=args.seed,
    )

    summary_cols = [args.raw_col, Q_COL, Q_ZERO_COL]
    if GAMMA_COL in out.columns:
        summary_cols.insert(1, GAMMA_COL)
    summary = pd.DataFrame([summary_for(out[col], col) for col in summary_cols])
    summary.to_csv(outdir / "support_roam_score_v5_quantile_summary.csv", index=False)

    champion = (
        out.groupby("support_champion_name", dropna=False)[[args.raw_col, Q_COL, Q_ZERO_COL]]
        .agg(["count", "mean", "median"])
    )
    champion.columns = ["_".join(col).strip("_") for col in champion.columns.to_flat_index()]
    champion = champion.reset_index().sort_values(f"{Q_ZERO_COL}_mean", ascending=False)
    champion.to_csv(outdir / "support_roam_score_v5_quantile_champion_means.csv", index=False)

    save_plots(out, outdir, args.raw_col)

    export_path = export_dir / args.out_name
    out.to_parquet(export_path, index=False)
    joblib.dump(transformers, outdir / "support_roam_score_v5_quantile_transformers.joblib")

    metadata = {
        "input": os.path.abspath(args.input),
        "export_path": str(export_path.resolve()),
        "raw_col": args.raw_col,
        "quantile_col": Q_COL,
        "zero_preserving_quantile_col": Q_ZERO_COL,
        "n_quantiles_requested": args.n_quantiles,
        "seed": args.seed,
        "important_note": (
            "These quantile columns are fitted globally for label exploration/model-input experiments. "
            "For strict validation, fit the transformer on the train split only and apply it to validation/test."
        ),
    }
    (outdir / "support_roam_score_v5_quantile_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"[Exported] {export_path.resolve()}")
    print(f"[Saved] {outdir.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
