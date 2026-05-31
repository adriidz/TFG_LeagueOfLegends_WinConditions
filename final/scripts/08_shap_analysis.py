#!/usr/bin/env python3
"""
08_shap_analysis.py -- SHAP explanations for the base raw HistGBT model.

The model is trained on ordinal-encoded categorical draft features. SHAP values
are therefore model-level associations, not causal champion effects.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = str(REPO_ROOT / "final" / "data" / "training" / "train.parquet")
DEFAULT_TEST = str(REPO_ROOT / "final" / "data" / "training" / "test.parquet")
DEFAULT_MODEL_DIR = str(REPO_ROOT / "final" / "models" / "gbt")
DEFAULT_OUTDIR = str(REPO_ROOT / "final" / "analysis" / "shap")

TARGET_COL = "support_roam_score"
MISSING_TOKEN = "__MISSING__"
ROLE_KEYS = ("top", "jungle", "middle", "bottom", "utility")
SIDES = ("ally", "enemy")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SHAP analysis for base HistGBT.")
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--test", default=DEFAULT_TEST)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--background-size", type=int, default=200)
    p.add_argument("--sample-size", type=int, default=2000)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_shap() -> Any:
    try:
        import shap  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'shap'. Install project requirements first "
            "(for example: python -m pip install -r requirements.txt)."
        ) from exc
    return shap


def encode_features(df: pd.DataFrame, feature_cols: List[str], encoder: Any) -> np.ndarray:
    raw = df[feature_cols].copy()
    for col in feature_cols:
        raw[col] = raw[col].fillna(MISSING_TOKEN).astype(str)
    return encoder.transform(raw)


def sample_frame(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or n >= len(df):
        return df.copy()
    return df.sample(n=n, random_state=seed).sort_index().copy()


def make_display_frame(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    display = pd.DataFrame(index=df.index)
    for col in feature_cols:
        name_col = col.replace("_champion_id", "_champion_name")
        if name_col in df.columns:
            display[col] = df[name_col].fillna(MISSING_TOKEN).astype(str)
        else:
            display[col] = df[col].fillna(MISSING_TOKEN).astype(str)
    return display


def build_explainer(
    shap: Any,
    model: Any,
    background: np.ndarray,
    feature_cols: List[str],
) -> Tuple[Any, str, Optional[str]]:
    try:
        explainer = shap.TreeExplainer(
            model,
            data=background,
            feature_names=feature_cols,
            model_output="raw",
        )
        return explainer, "TreeExplainer", None
    except Exception as exc:  # depends on shap/sklearn internals
        masker = shap.maskers.Independent(background)
        explainer = shap.PermutationExplainer(
            model.predict,
            masker,
            feature_names=feature_cols,
        )
        return explainer, "PermutationExplainer", repr(exc)


def build_permutation_explainer(
    shap: Any,
    model: Any,
    background: np.ndarray,
    reason: str,
) -> Tuple[Any, str, str]:
    masker = shap.maskers.Independent(background)
    explainer = shap.PermutationExplainer(model.predict, masker)
    return explainer, "PermutationExplainer", reason


def compute_shap_values(
    explainer: Any,
    explainer_type: str,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray | float]:
    if explainer_type == "TreeExplainer":
        values = explainer.shap_values(X, check_additivity=False)
        expected = explainer.expected_value
        return np.asarray(values, dtype=np.float64), expected

    max_evals = 2 * X.shape[1] + 1
    explanation = explainer(X, max_evals=max_evals, silent=True)
    return np.asarray(explanation.values, dtype=np.float64), explanation.base_values


def scalar_expected(expected: Any) -> float:
    arr = np.asarray(expected, dtype=np.float64)
    return float(arr.reshape(-1)[0])


def expected_vector(expected: Any, n_rows: int) -> np.ndarray:
    arr = np.asarray(expected, dtype=np.float64)
    if arr.size == 1:
        return np.full(n_rows, float(arr.reshape(-1)[0]), dtype=np.float64)
    return arr.reshape(-1).astype(np.float64)


def additivity_error(predictions: np.ndarray, expected: Any, values: np.ndarray) -> float:
    shap_pred = expected_vector(expected, len(predictions)) + values.sum(axis=1)
    return float(np.max(np.abs(predictions - shap_pred)))


def save_global_importance(
    values: np.ndarray,
    feature_cols: List[str],
    outdir: Path,
) -> pd.DataFrame:
    imp = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_abs_shap": np.mean(np.abs(values), axis=0),
            "mean_shap": np.mean(values, axis=0),
            "std_shap": np.std(values, axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(outdir / "shap_global_importance.csv", index=False)
    return imp


def save_summary_plots(
    shap: Any,
    values: np.ndarray,
    X: np.ndarray,
    feature_cols: List[str],
    outdir: Path,
    top_n: int,
) -> None:
    max_display = min(top_n, len(feature_cols))

    plt.figure(figsize=(10, max(5, 0.28 * max_display + 1.5)))
    shap.summary_plot(
        values,
        X,
        feature_names=feature_cols,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(outdir / "shap_summary_bar.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, max(5, 0.28 * max_display + 1.5)))
    shap.summary_plot(
        values,
        X,
        feature_names=feature_cols,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(outdir / "shap_summary_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close()


def save_categorical_dependence(
    values: np.ndarray,
    display_df: pd.DataFrame,
    feature_cols: List[str],
    feature: str,
    outdir: Path,
    top_n: int,
) -> None:
    if feature not in feature_cols:
        return
    idx = feature_cols.index(feature)
    tmp = pd.DataFrame(
        {
            "category": display_df[feature].astype(str).to_numpy(),
            "shap_value": values[:, idx],
        }
    )
    grouped = (
        tmp.groupby("category")["shap_value"]
        .agg(
            mean_shap="mean",
            mean_abs_shap=lambda s: float(np.mean(np.abs(s))),
            n="count",
        )
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .sort_values("mean_shap", ascending=True)
    )
    fig_h = max(4.0, 0.35 * len(grouped) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    colors = np.where(grouped["mean_shap"].to_numpy() >= 0, "#2f80ed", "#eb5757")
    ax.barh(grouped.index.astype(str), grouped["mean_shap"].to_numpy(), color=colors, alpha=0.85)
    ax.axvline(0.0, color="#222222", linewidth=0.8)
    ax.set_xlabel("Mean SHAP contribution")
    ax.set_ylabel(feature)
    ax.set_title(f"Categorical SHAP dependence: {feature}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / f"shap_dependence_{feature}.png", dpi=180)
    plt.close(fig)


def select_local_cases(df: pd.DataFrame) -> pd.DataFrame:
    candidates: List[Tuple[str, pd.Series]] = []

    high_pred_low_actual = df[
        (df["prediction"] >= df["prediction"].quantile(0.85))
        & (df["actual"] <= df["actual"].quantile(0.20))
    ]
    if not high_pred_low_actual.empty:
        candidates.append(
            (
                "high_pred_low_actual",
                high_pred_low_actual.sort_values("abs_error", ascending=False).iloc[0],
            )
        )

    low_pred_high_actual = df[
        (df["prediction"] <= df["prediction"].quantile(0.20))
        & (df["actual"] >= df["actual"].quantile(0.85))
    ]
    if not low_pred_high_actual.empty:
        candidates.append(
            (
                "low_pred_high_actual",
                low_pred_high_actual.sort_values("abs_error", ascending=False).iloc[0],
            )
        )

    accurate = df[df["abs_error"] <= df["abs_error"].quantile(0.05)]
    if not accurate.empty:
        mid_pred = float(df["prediction"].median())
        accurate = accurate.assign(_distance_to_mid=(accurate["prediction"] - mid_pred).abs())
        candidates.append(("high_accuracy", accurate.sort_values("_distance_to_mid").iloc[0]))

    roamers = df[df["ally_utility_champion_name"].isin(["Bard", "Pyke", "Rakan", "Thresh", "Nautilus", "Leona"])]
    if not roamers.empty:
        candidates.append(("roam_support", roamers.sort_values("abs_error", ascending=False).iloc[0]))

    if not candidates:
        candidates.append(("largest_abs_error", df.sort_values("abs_error", ascending=False).iloc[0]))

    rows: List[pd.Series] = []
    seen = set()
    for label, row in candidates:
        key = (row["match_id"], row["team_id"])
        if key in seen:
            continue
        seen.add(key)
        row = row.copy()
        row["case_label"] = label
        rows.append(row)

    out = pd.DataFrame(rows)
    cols = ["case_label"] + [c for c in out.columns if c != "case_label"]
    return out[cols]


def save_waterfalls(
    shap: Any,
    values: np.ndarray,
    expected: Any,
    sample_df: pd.DataFrame,
    display_df: pd.DataFrame,
    feature_cols: List[str],
    cases: pd.DataFrame,
    outdir: Path,
) -> List[Dict[str, Any]]:
    base = scalar_expected(expected)
    rows: List[Dict[str, Any]] = []
    index_to_pos = {idx: pos for pos, idx in enumerate(sample_df.index)}

    for case_no, (_, case) in enumerate(cases.iterrows(), start=1):
        original_idx = case.name
        pos = index_to_pos.get(original_idx)
        if pos is None:
            continue
        explanation = shap.Explanation(
            values=values[pos],
            base_values=base,
            data=display_df.iloc[pos].to_numpy(),
            feature_names=feature_cols,
        )
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=12, show=False)
        fig.tight_layout()
        filename = f"shap_waterfall_case_{case_no:02d}_{case['case_label']}.png"
        fig.savefig(outdir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
        rows.append(
            {
                "case_label": case["case_label"],
                "waterfall_plot": filename,
                "match_id": case["match_id"],
                "team_id": case["team_id"],
                "side": case["side"],
                "patch": case.get("patch", ""),
                "ally_utility_champion_name": case.get("ally_utility_champion_name", ""),
                "ally_bottom_champion_name": case.get("ally_bottom_champion_name", ""),
                "enemy_utility_champion_name": case.get("enemy_utility_champion_name", ""),
                "enemy_bottom_champion_name": case.get("enemy_bottom_champion_name", ""),
                "prediction": float(case["prediction"]),
                "actual": float(case["actual"]),
                "abs_error": float(case["abs_error"]),
                "shap_sum_prediction": float(base + values[pos].sum()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    shap = load_shap()

    model_dir = Path(args.model_dir)
    model = joblib.load(model_dir / "gbt_model_raw.joblib")
    preprocess = joblib.load(model_dir / "preprocess.joblib")
    encoder = preprocess["encoder"]
    feature_cols: List[str] = preprocess["feature_columns"]

    df_train = pd.read_parquet(args.train)
    df_test_full = pd.read_parquet(args.test)
    df_test = sample_frame(df_test_full, args.sample_size, args.seed)
    df_background = sample_frame(df_train, args.background_size, args.seed)

    X_background = encode_features(df_background, feature_cols, encoder)
    X_sample = encode_features(df_test, feature_cols, encoder)
    display_df = make_display_frame(df_test, feature_cols)

    print(
        f"[Data] train={len(df_train):,}  test_sample={len(df_test):,}  "
        f"background={len(df_background):,}  features={len(feature_cols)}"
    )
    explainer, explainer_type, fallback_reason = build_explainer(
        shap, model, X_background, feature_cols
    )
    print(f"[SHAP] explainer={explainer_type}")
    if fallback_reason:
        print(f"[SHAP] TreeExplainer fallback reason: {fallback_reason}")

    values, expected = compute_shap_values(explainer, explainer_type, X_sample)
    predictions = model.predict(X_sample)
    actual = df_test[TARGET_COL].to_numpy(dtype=np.float64)
    additivity_max_abs_error = additivity_error(predictions, expected, values)
    additivity_tolerance = 1e-6

    if explainer_type == "TreeExplainer" and additivity_max_abs_error > additivity_tolerance:
        fallback_reason = (
            "TreeExplainer produced non-additive values for this sklearn "
            f"HistGradientBoostingRegressor (max_abs_error={additivity_max_abs_error:.8f})."
        )
        print(f"[SHAP] {fallback_reason}")
        explainer, explainer_type, fallback_reason = build_permutation_explainer(
            shap, model, X_background, fallback_reason
        )
        print("[SHAP] explainer=PermutationExplainer")
        values, expected = compute_shap_values(explainer, explainer_type, X_sample)
        additivity_max_abs_error = additivity_error(predictions, expected, values)

    importance_df = save_global_importance(values, feature_cols, outdir)
    save_summary_plots(shap, values, X_sample, feature_cols, outdir, args.top_n)
    save_categorical_dependence(
        values, display_df, feature_cols, "ally_utility_champion_id", outdir, args.top_n
    )
    save_categorical_dependence(
        values, display_df, feature_cols, "ally_bottom_champion_id", outdir, args.top_n
    )

    sample_eval = df_test.copy()
    sample_eval["prediction"] = predictions
    sample_eval["actual"] = actual
    sample_eval["signed_error"] = predictions - actual
    sample_eval["abs_error"] = np.abs(sample_eval["signed_error"])
    local_cases = select_local_cases(sample_eval)
    waterfall_rows = save_waterfalls(
        shap, values, expected, sample_eval, display_df, feature_cols, local_cases, outdir
    )
    pd.DataFrame(waterfall_rows).to_csv(outdir / "shap_local_top_cases.csv", index=False)

    meta: Dict[str, Any] = {
        "model_path": str((model_dir / "gbt_model_raw.joblib").resolve()),
        "preprocess_path": str((model_dir / "preprocess.joblib").resolve()),
        "train_path": str(Path(args.train).resolve()),
        "test_path": str(Path(args.test).resolve()),
        "outdir": str(outdir.resolve()),
        "target": TARGET_COL,
        "explainer": explainer_type,
        "tree_explainer_fallback_reason": fallback_reason,
        "seed": args.seed,
        "background_size": int(len(df_background)),
        "sample_size": int(len(df_test)),
        "top_n": args.top_n,
        "n_features": len(feature_cols),
        "expected_value": scalar_expected(expected),
        "additivity_max_abs_error": additivity_max_abs_error,
        "additivity_tolerance": additivity_tolerance,
        "additivity_checked": bool(math.isfinite(additivity_max_abs_error)),
        "additivity_passed": bool(additivity_max_abs_error <= additivity_tolerance),
        "top_features": importance_df.head(args.top_n).to_dict(orient="records"),
        "note": (
            "SHAP values explain the fitted model over ordinal-encoded categorical "
            "draft features. Interpret them as associations in model space, not "
            "causal effects of champion choices."
        ),
    }
    (outdir / "shap_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[Additivity] max_abs_error={additivity_max_abs_error:.8f}")
    print(f"[Saved] {outdir.resolve()}")


if __name__ == "__main__":
    main()
