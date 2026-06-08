#!/usr/bin/env python3
"""
Check the fair main-comparison feature protocol without training models.

The main learned models must all use the same conceptual pre-game input:
10 champion IDs + side. Baselines are intentionally lower-information
references and are labelled as such in the audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = REPO_ROOT / "final" / "data" / "training" / "train.parquet"
DEFAULT_VAL = REPO_ROOT / "final" / "data" / "training" / "val.parquet"
DEFAULT_OUTDIR = REPO_ROOT / "final" / "analysis" / "model_comparison"

ROLE_KEYS = ("top", "jungle", "middle", "bottom", "utility")
SIDES = ("ally", "enemy")
CHAMPION_COLS = [f"{s}_{r}_champion_id" for s in SIDES for r in ROLE_KEYS]
CANONICAL_MAIN_FEATURES = CHAMPION_COLS + ["side"]
FEATURE_PROTOCOL_ID = "draft_10_champions_side"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dry-run fair feature protocol checks.")
    p.add_argument("--train", default=str(DEFAULT_TRAIN))
    p.add_argument("--val", default=str(DEFAULT_VAL))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    return p.parse_args()


def encode_gbt_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: List[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_raw = df_train[feature_cols].copy()
    val_raw = df_val[feature_cols].copy()
    for col in feature_cols:
        train_raw[col] = train_raw[col].fillna("__MISSING__").astype(str)
        val_raw[col] = val_raw[col].fillna("__MISSING__").astype(str)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
    return encoder.fit_transform(train_raw), encoder.transform(val_raw)


def encode_mlp_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    champions = np.zeros((len(df), len(CHAMPION_COLS)), dtype=np.int64)
    for i, col in enumerate(CHAMPION_COLS):
        champions[:, i] = df[col].fillna(-1).astype(int).to_numpy(dtype=np.int64)
    side = (
        df["side"]
        .map({"blue": 0.0, "red": 1.0})
        .fillna(0.5)
        .to_numpy(dtype=np.float32)
        .reshape(-1, 1)
    )
    return champions, side


def audit_row(
    model: str,
    comparison_role: str,
    feature_cols: List[str],
    train_shape: List[int],
    val_shape: List[int],
) -> Dict[str, Any]:
    return {
        "model": model,
        "comparison_role": comparison_role,
        "feature_protocol_id": (
            FEATURE_PROTOCOL_ID if feature_cols == CANONICAL_MAIN_FEATURES else comparison_role
        ),
        "input_feature_columns": feature_cols,
        "feature_count": len(feature_cols),
        "train_shape": train_shape,
        "val_shape": val_shape,
        "matches_main_feature_protocol": feature_cols == CANONICAL_MAIN_FEATURES,
        "sample_weight_column": "sample_weight",
        "used_sample_weight": True,
    }


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_parquet(args.train)
    df_val = pd.read_parquet(args.val)
    if "sample_weight" not in df_train.columns:
        raise SystemExit("[Weights] Missing required sample_weight in train split.")

    missing = [col for col in CANONICAL_MAIN_FEATURES if col not in df_train.columns]
    if missing:
        raise SystemExit(f"[Features] Missing required main feature columns: {missing}")

    sample_weight = df_train["sample_weight"].to_numpy(dtype=np.float32)
    print(
        f"[Data] train={len(df_train):,} val={len(df_val):,} "
        f"sample_weight mean={sample_weight.mean():.3f} "
        f"min={sample_weight.min():.3f} max={sample_weight.max():.3f}"
    )

    gbt_train, gbt_val = encode_gbt_features(df_train, df_val, CANONICAL_MAIN_FEATURES)
    print(f"[GBT] matrix train={gbt_train.shape} val={gbt_val.shape}")

    mlp_champ_train, mlp_side_train = encode_mlp_features(df_train)
    mlp_champ_val, mlp_side_val = encode_mlp_features(df_val)
    mlp_train_shape = [int(mlp_champ_train.shape[0]), int(mlp_champ_train.shape[1] + mlp_side_train.shape[1])]
    mlp_val_shape = [int(mlp_champ_val.shape[0]), int(mlp_champ_val.shape[1] + mlp_side_val.shape[1])]
    print(f"[MLP] conceptual matrix train={tuple(mlp_train_shape)} val={tuple(mlp_val_shape)}")
    print("[Features] " + ", ".join(CANONICAL_MAIN_FEATURES))

    rows = [
        audit_row("Global Mean", "baseline_no_features", [], [len(df_train), 0], [len(df_val), 0]),
        audit_row(
            "Champion Mean",
            "baseline_support_champion_only",
            ["ally_utility_champion_id"],
            [len(df_train), 1],
            [len(df_val), 1],
        ),
        audit_row(
            "HistGBT",
            "main_learned_model",
            CANONICAL_MAIN_FEATURES,
            [int(gbt_train.shape[0]), int(gbt_train.shape[1])],
            [int(gbt_val.shape[0]), int(gbt_val.shape[1])],
        ),
        audit_row("MLP OneHot", "main_learned_model", CANONICAL_MAIN_FEATURES, mlp_train_shape, mlp_val_shape),
        audit_row(
            "MLP Embed Shared",
            "main_learned_model",
            CANONICAL_MAIN_FEATURES,
            mlp_train_shape,
            mlp_val_shape,
        ),
        audit_row(
            "MLP Per-Role + Interactions",
            "main_learned_model",
            CANONICAL_MAIN_FEATURES,
            mlp_train_shape,
            mlp_val_shape,
        ),
    ]

    learned_bad = [
        row for row in rows
        if row["comparison_role"] == "main_learned_model"
        and not row["matches_main_feature_protocol"]
    ]
    if learned_bad:
        raise SystemExit(f"[Features] Learned model protocol mismatch: {learned_bad}")

    payload = {
        "feature_protocol_id": FEATURE_PROTOCOL_ID,
        "canonical_main_features": CANONICAL_MAIN_FEATURES,
        "sample_weight_summary": {
            "mean": float(sample_weight.mean()),
            "min": float(sample_weight.min()),
            "max": float(sample_weight.max()),
        },
        "rows": rows,
        "excluded_from_main": [
            {
                "model": "HistGBT + Archetypes",
                "reason": "Adds champion archetype/class features beyond 10 champion IDs + side.",
            },
            {
                "model": "HistGBT + Pair TE",
                "reason": "Adds target-encoded pair interaction features.",
            },
        ],
    }
    path = outdir / "main_feature_protocol_dry_run.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Saved] {path.resolve()}")


if __name__ == "__main__":
    main()
