#!/usr/bin/env python3
"""
21_clean_only_gbt_experiment.py -- Train HistGBT on clean rows only.

This is a secondary diagnostic experiment for the clean-vs-chaotic analysis.
It does not replace the main model comparison protocol. The script trains a
HistGBT model on train rows where chaos_flag is False, then evaluates both the
main final HistGBT and the clean-only HistGBT on the same held-out test split
for all, clean, and chaotic subsets.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = REPO_ROOT / "final" / "data" / "training"
DEFAULT_TRAIN = TRAINING_DIR / "train.parquet"
DEFAULT_TEST = TRAINING_DIR / "test.parquet"
DEFAULT_MAIN_MODEL_DIR = REPO_ROOT / "final" / "models" / "gbt"
DEFAULT_CLEAN_MODEL_DIR = REPO_ROOT / "final" / "models" / "gbt_clean_only"
DEFAULT_OUTDIR = REPO_ROOT / "final" / "analysis" / "clean_vs_chaotic"

TARGET_COL = "support_roam_score"
FEATURE_PROTOCOL_ID = "draft_10_champions_side"
FEATURE_COLS = [
    "ally_top_champion_id",
    "ally_jungle_champion_id",
    "ally_middle_champion_id",
    "ally_bottom_champion_id",
    "ally_utility_champion_id",
    "enemy_top_champion_id",
    "enemy_jungle_champion_id",
    "enemy_middle_champion_id",
    "enemy_bottom_champion_id",
    "enemy_utility_champion_id",
    "side",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate clean-only HistGBT experiment.")
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--test", default=str(DEFAULT_TEST))
    parser.add_argument("--main-model-dir", default=str(DEFAULT_MAIN_MODEL_DIR))
    parser.add_argument("--clean-model-dir", default=str(DEFAULT_CLEAN_MODEL_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def validate_chaos_flag(df: pd.DataFrame, split_name: str) -> None:
    if "chaos_flag" not in df.columns:
        raise SystemExit(f"[{split_name}] Missing chaos_flag column.")
    n_nan = int(df["chaos_flag"].isna().sum())
    if n_nan:
        raise SystemExit(f"[{split_name}] chaos_flag has {n_nan} NaN rows.")


def encode_features(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, OrdinalEncoder]:
    train_raw = df_train[FEATURE_COLS].copy()
    eval_raw = df_eval[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        train_raw[col] = train_raw[col].fillna("__MISSING__").astype(str)
        eval_raw[col] = eval_raw[col].fillna("__MISSING__").astype(str)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
    return encoder.fit_transform(train_raw), encoder.transform(eval_raw), encoder


def train_clean_seed(
    df_train_clean: pd.DataFrame,
    df_test: pd.DataFrame,
    seed: int,
    model_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "gbt_model_raw.joblib"
    preprocess_path = model_dir / "preprocess.joblib"
    config_path = model_dir / "model_config.json"
    if model_path.exists() and preprocess_path.exists() and config_path.exists() and not args.force_retrain:
        print(f"[Skip train] seed={seed} existing clean-only artifact: {model_dir}")
    else:
        print(f"[Train clean-only] seed={seed} rows={len(df_train_clean):,}")
        x_train, _, encoder = encode_features(df_train_clean, df_test)
        y_train = df_train_clean[TARGET_COL].to_numpy(dtype=np.float32)
        sample_weight = df_train_clean["sample_weight"].to_numpy(dtype=np.float32)

        model = HistGradientBoostingRegressor(
            max_iter=args.max_iter,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            min_samples_leaf=args.min_samples_leaf,
            max_leaf_nodes=args.max_leaf_nodes,
            categorical_features=[True] * len(FEATURE_COLS),
            random_state=seed,
            verbose=1,
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
        joblib.dump(model, model_path)
        joblib.dump(
            {
                "encoder": encoder,
                "feature_columns": FEATURE_COLS,
                "feature_set": "main_clean_train_only",
                "feature_protocol_id": FEATURE_PROTOCOL_ID,
                "sample_weight_column": "sample_weight",
                "used_sample_weight": True,
            },
            preprocess_path,
        )
        config = {
            "model_type": "hist_gbt",
            "training_subset": "clean_only",
            "feature_set": "main",
            "feature_protocol_id": FEATURE_PROTOCOL_ID,
            "feature_columns": FEATURE_COLS,
            "sample_weight_column": "sample_weight",
            "used_sample_weight": True,
            "max_iter": args.max_iter,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "min_samples_leaf": args.min_samples_leaf,
            "max_leaf_nodes": args.max_leaf_nodes,
            "seed": seed,
            "n_train": int(len(df_train_clean)),
            "n_train_chaotic": 0,
        }
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        (model_dir / "feature_audit.json").write_text(
            json.dumps(
                {
                    "model": "HistGBT clean-only",
                    "feature_protocol_id": FEATURE_PROTOCOL_ID,
                    "input_feature_columns": FEATURE_COLS,
                    "feature_count": len(FEATURE_COLS),
                    "matrix_shape_train": [int(x_train.shape[0]), int(x_train.shape[1])],
                    "sample_weight_column": "sample_weight",
                    "used_sample_weight": True,
                    "sample_weight_summary": {
                        "mean": float(sample_weight.mean()),
                        "min": float(sample_weight.min()),
                        "max": float(sample_weight.max()),
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return load_seed_manifest(model_dir)


def load_seed_manifest(model_dir: Path) -> Dict[str, Any]:
    config_path = model_dir / "model_config.json"
    preprocess_path = model_dir / "preprocess.joblib"
    model_path = model_dir / "gbt_model_raw.joblib"
    if not config_path.exists() or not preprocess_path.exists() or not model_path.exists():
        raise SystemExit(f"Incomplete HistGBT artifact: {model_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "run_dir": str(model_dir.resolve()),
        "model_path": str(model_path.resolve()),
        "preprocess_path": str(preprocess_path.resolve()),
        "config_path": str(config_path.resolve()),
        "seed": config.get("seed"),
        "training_subset": config.get("training_subset", "all_weighted"),
        "feature_protocol_id": config.get("feature_protocol_id"),
        "feature_columns": config.get("feature_columns", FEATURE_COLS),
        "used_sample_weight": bool(config.get("used_sample_weight")),
    }


def seed_dirs_for(model_root: Path, seeds: List[int]) -> List[Path]:
    dirs = [model_root / f"seed{seed}" for seed in seeds]
    missing = [str(path) for path in dirs if not path.exists()]
    if missing:
        raise SystemExit(f"Missing model seed dirs: {missing}")
    return dirs


def predict_seed_ensemble(seed_dirs: List[Path], df_test: pd.DataFrame) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    pred_by_seed: List[np.ndarray] = []
    manifests: List[Dict[str, Any]] = []
    reference_cols: Optional[List[str]] = None

    for seed_dir in seed_dirs:
        manifest = load_seed_manifest(seed_dir)
        if manifest["feature_protocol_id"] != FEATURE_PROTOCOL_ID:
            raise SystemExit(f"{seed_dir} has unexpected feature protocol: {manifest['feature_protocol_id']}")
        if manifest["used_sample_weight"] is not True:
            raise SystemExit(f"{seed_dir} was not trained with sample_weight.")

        preprocess = joblib.load(seed_dir / "preprocess.joblib")
        feature_cols = list(preprocess["feature_columns"])
        if reference_cols is None:
            reference_cols = feature_cols
        elif feature_cols != reference_cols:
            raise SystemExit("Feature columns differ across seed artifacts.")

        x_raw = df_test[feature_cols].copy()
        for col in feature_cols:
            x_raw[col] = x_raw[col].fillna("__MISSING__").astype(str)
        x_test = preprocess["encoder"].transform(x_raw)
        model = joblib.load(seed_dir / "gbt_model_raw.joblib")
        pred_by_seed.append(model.predict(x_test).astype(np.float64))
        manifests.append(manifest)

    return np.vstack(pred_by_seed).mean(axis=0), manifests


def metrics_for(y: np.ndarray, pred: np.ndarray, mask: np.ndarray, label: str, model_label: str) -> Dict[str, Any]:
    yt = y[mask]
    yp = pred[mask]
    mse = float(np.mean((yt - yp) ** 2))
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    sp = spearmanr(yt, yp, nan_policy="omit")
    return {
        "model": model_label,
        "subset": label,
        "n": int(len(yt)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "spearman": float(sp.correlation) if sp.correlation is not None else float("nan"),
        "mae": float(np.mean(np.abs(yt - yp))),
        "rmse": math.sqrt(mse),
        "target_mean": float(yt.mean()),
        "pred_std": float(yp.std()),
    }


def evaluate_model(model_label: str, pred: np.ndarray, df_test: pd.DataFrame) -> List[Dict[str, Any]]:
    y = df_test[TARGET_COL].to_numpy(dtype=np.float64)
    chaos = df_test["chaos_flag"].to_numpy(dtype=bool)
    masks = [
        ("all", np.ones(len(df_test), dtype=bool)),
        ("clean", ~chaos),
        ("chaotic", chaos),
    ]
    rows = [metrics_for(y, pred, mask, subset, model_label) for subset, mask in masks]
    if rows[1]["n"] + rows[2]["n"] != rows[0]["n"]:
        raise SystemExit(f"[{model_label}] clean + chaotic != all")
    return rows


def write_markdown(outpath: Path, rows: List[Dict[str, Any]], checks: Dict[str, Any]) -> None:
    lines = [
        "# Clean-Only Training Experiment",
        "",
        "This is a secondary diagnostic, not the main model comparison table.",
        "",
        "Both model sources are evaluated on the same held-out test split and the same feature protocol.",
        "",
        "Validation checks:",
        "",
        f"- `clean + chaotic == all`: {checks['n_clean']:,} + {checks['n_chaotic']:,} = {checks['n_all']:,}",
        f"- `chaos_flag` NaN count in train/test: {checks['train_chaos_nan_count']} / {checks['test_chaos_nan_count']}",
        f"- Clean-only train rows: {checks['n_train_clean']:,} of {checks['n_train_all']:,}",
        f"- Feature protocol: `{FEATURE_PROTOCOL_ID}`",
        f"- Seeds: {', '.join(str(seed) for seed in checks['seeds'])}",
        "",
        "| model | subset | n | R2 | Spearman | MAE | RMSE | target_mean | pred_std |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['subset']} | {row['n']:,} | "
            f"{row['r2']:.4f} | {row['spearman']:.4f} | {row['mae']:.4f} | "
            f"{row['rmse']:.4f} | {row['target_mean']:.4f} | {row['pred_std']:.4f} |"
        )
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_path = Path(args.train)
    test_path = Path(args.test)
    clean_model_root = Path(args.clean_model_dir)
    main_model_root = Path(args.main_model_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    validate_chaos_flag(df_train, "train")
    validate_chaos_flag(df_test, "test")
    if "sample_weight" not in df_train.columns:
        raise SystemExit("[train] Missing sample_weight column.")

    df_train_clean = df_train[~df_train["chaos_flag"].astype(bool)].copy()
    for seed in args.seeds:
        train_clean_seed(
            df_train_clean=df_train_clean,
            df_test=df_test,
            seed=seed,
            model_dir=clean_model_root / f"seed{seed}",
            args=args,
        )

    main_pred, main_manifests = predict_seed_ensemble(seed_dirs_for(main_model_root, args.seeds), df_test)
    clean_pred, clean_manifests = predict_seed_ensemble(seed_dirs_for(clean_model_root, args.seeds), df_test)

    rows = []
    rows.extend(evaluate_model("HistGBT final weighted-train", main_pred, df_test))
    rows.extend(evaluate_model("HistGBT clean-only train", clean_pred, df_test))

    checks = {
        "n_train_all": int(len(df_train)),
        "n_train_clean": int(len(df_train_clean)),
        "n_train_chaotic_excluded": int(df_train["chaos_flag"].astype(bool).sum()),
        "n_all": int(len(df_test)),
        "n_clean": int((~df_test["chaos_flag"].astype(bool)).sum()),
        "n_chaotic": int(df_test["chaos_flag"].astype(bool).sum()),
        "clean_plus_chaotic_equals_all": bool(
            int((~df_test["chaos_flag"].astype(bool)).sum())
            + int(df_test["chaos_flag"].astype(bool).sum())
            == len(df_test)
        ),
        "train_chaos_nan_count": int(df_train["chaos_flag"].isna().sum()),
        "test_chaos_nan_count": int(df_test["chaos_flag"].isna().sum()),
        "feature_protocol_id": FEATURE_PROTOCOL_ID,
        "seeds": list(args.seeds),
        "main_model_artifacts": main_manifests,
        "clean_only_model_artifacts": clean_manifests,
    }

    pd.DataFrame(rows).to_csv(outdir / "clean_only_training_experiment.csv", index=False)
    (outdir / "clean_only_training_experiment_checks.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(outdir / "clean_only_training_experiment.md", rows, checks)

    print("[Saved] clean-only experiment outputs:")
    print(f"  {outdir / 'clean_only_training_experiment.csv'}")
    for row in rows:
        print(
            f"  {row['model']} / {row['subset']}: "
            f"R2={row['r2']:.4f} Spearman={row['spearman']:.4f} n={row['n']:,}"
        )


if __name__ == "__main__":
    main()
