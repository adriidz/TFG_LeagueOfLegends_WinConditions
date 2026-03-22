#!/usr/bin/env python3
"""
train_jungle_presence_model_v2.py

V2 del entrenamiento para jungle_presence_label.

Qué añade respecto a v1:
- Ablations:
    * majority_baseline
    * jungle_only_logreg
    * jungle_mid_support_logreg
    * full_draft_logreg
- Modelos no lineales:
    * random_forest_jungle_only
    * random_forest_full_draft
- Mismo esquema de entrada:
    parquet con una fila por (match_id, team_id)

Notas:
- Pensado para CPU.
- No requiere GPU/CUDA.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COL = "jungle_presence_label"
POSITIVE_LABEL = "map_presence"
NEGATIVE_LABEL = "farm_oriented"

DEFAULT_FULL_FEATURES = [
    "ally_top_champion_name",
    "ally_jungle_champion_name",
    "ally_middle_champion_name",
    "ally_bottom_champion_name",
    "ally_utility_champion_name",
    "enemy_top_champion_name",
    "enemy_jungle_champion_name",
    "enemy_middle_champion_name",
    "enemy_bottom_champion_name",
    "enemy_utility_champion_name",
    "side",
    "patch",
]

DEFAULT_JUNGLE_ONLY_FEATURES = [
    "ally_jungle_champion_name",
    "side",
    "patch",
]

DEFAULT_JUNGLE_MID_SUPPORT_FEATURES = [
    "ally_jungle_champion_name",
    "ally_middle_champion_name",
    "ally_utility_champion_name",
    "enemy_jungle_champion_name",
    "enemy_middle_champion_name",
    "enemy_utility_champion_name",
    "side",
    "patch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena modelos v2 para jungle presence desde draft."
    )
    parser.add_argument("--input-path", default=os.path.join("Data_training", "model_input_jungle.parquet"), help="Parquet de entrada.")
    parser.add_argument("--output-dir", default=os.path.join("Models", "jungle_presence_random_forest"), help="Directorio de salida.")
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument("--positive-label", default=POSITIVE_LABEL)
    parser.add_argument("--negative-label", default=NEGATIVE_LABEL)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--class-weight-balanced", action="store_true")
    parser.add_argument(
        "--full-features", nargs="*", default=DEFAULT_FULL_FEATURES
    )
    parser.add_argument(
        "--jungle-only-features", nargs="*", default=DEFAULT_JUNGLE_ONLY_FEATURES
    )
    parser.add_argument(
        "--jungle-mid-support-features",
        nargs="*",
        default=DEFAULT_JUNGLE_MID_SUPPORT_FEATURES,
    )
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=300,
        help="Número de árboles para RandomForest (default: 300).",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=None,
        help="Profundidad máxima de RandomForest (default: None).",
    )
    parser.add_argument(
        "--rf-min-samples-leaf",
        type=int,
        default=1,
        help="min_samples_leaf para RandomForest (default: 1).",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo de entrada: {path}")
    if not path.endswith(".parquet"):
        raise ValueError("Solo se admite .parquet en esta versión.")
    return pd.read_parquet(path)


def filter_target_classes(
    df: pd.DataFrame,
    target_col: str,
    positive_label: str,
    negative_label: str,
) -> pd.DataFrame:
    keep = {positive_label, negative_label}
    out = df[df[target_col].isin(keep)].copy()
    if out.empty:
        raise ValueError(f"No quedan filas tras filtrar {target_col} por {sorted(keep)}.")
    return out


def select_existing_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    return [c for c in columns if c in df.columns]


def encode_target(y: pd.Series, positive_label: str, negative_label: str) -> np.ndarray:
    mapping = {negative_label: 0, positive_label: 1}
    return y.map(mapping).astype(int).to_numpy()


def build_preprocessor(categorical_features: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def build_logreg_pipeline(
    categorical_features: List[str],
    class_weight_balanced: bool = False,
) -> Pipeline:
    preprocessor = build_preprocessor(categorical_features)
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced" if class_weight_balanced else None,
        solver="lbfgs",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clf),
        ]
    )


def build_random_forest_pipeline(
    categorical_features: List[str],
    n_estimators: int,
    max_depth,
    min_samples_leaf: int,
    random_state: int,
    class_weight_balanced: bool = False,
) -> Pipeline:
    preprocessor = build_preprocessor(categorical_features)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced" if class_weight_balanced else None,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clf),
        ]
    )


def evaluate_binary_model(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    positive_label: str,
    negative_label: str,
) -> Tuple[Dict, pd.DataFrame]:
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "n_test": int(len(y_test)),
        "positive_rate_test": float(np.mean(y_test)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba)) if proba is not None else None
    except Exception:
        metrics["roc_auc"] = None

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = {
        negative_label: {
            negative_label: int(cm[0, 0]),
            positive_label: int(cm[0, 1]),
        },
        positive_label: {
            negative_label: int(cm[1, 0]),
            positive_label: int(cm[1, 1]),
        },
    }
    metrics["classification_report"] = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[negative_label, positive_label],
        output_dict=True,
        zero_division=0,
    )

    pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    if proba is not None:
        pred_df["p_map_presence"] = proba
    return metrics, pred_df


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_dummy_baseline(
    y_train: np.ndarray,
    y_test: np.ndarray,
    positive_label: str,
    negative_label: str,
):
    clf = DummyClassifier(strategy="most_frequent")
    X_train = pd.DataFrame({"dummy": np.zeros(len(y_train), dtype=int)})
    X_test = pd.DataFrame({"dummy": np.zeros(len(y_test), dtype=int)})
    clf.fit(X_train, y_train)
    metrics, pred_df = evaluate_binary_model(
        clf, X_test, y_test, positive_label, negative_label
    )
    return metrics, pred_df, clf


def run_model(
    model_name: str,
    pipeline: Pipeline,
    feature_cols: List[str],
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    positive_label: str,
    negative_label: str,
):
    print(f"Entrenando {model_name}...")
    pipeline.fit(train_df[feature_cols], y_train)
    metrics, pred_df = evaluate_binary_model(
        pipeline,
        test_df[feature_cols],
        y_test,
        positive_label,
        negative_label,
    )
    metrics["features"] = feature_cols
    return metrics, pred_df, pipeline


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    print("Cargando parquet...")
    df = load_dataset(args.input_path)
    print(f"Filas iniciales: {len(df)}")

    if args.target_col not in df.columns:
        raise ValueError(f"No existe la columna objetivo: {args.target_col}")

    print("Filtrando clases extremas...")
    df = filter_target_classes(
        df,
        target_col=args.target_col,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
    )
    print(f"Filas tras filtrar ambiguous: {len(df)}")

    id_cols = [c for c in ["match_id", "team_id"] if c in df.columns]

    feature_sets = {
        "jungle_only": select_existing_columns(df, args.jungle_only_features),
        "jungle_mid_support": select_existing_columns(df, args.jungle_mid_support_features),
        "full_draft": select_existing_columns(df, args.full_features),
    }

    for name, feats in feature_sets.items():
        if not feats:
            raise ValueError(f"No se encontró ninguna feature para {name}.")
        print(f"Features {name}: {feats}")

    y = encode_target(
        df[args.target_col],
        positive_label=args.positive_label,
        negative_label=args.negative_label,
    )

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    y_train = encode_target(
        train_df[args.target_col], args.positive_label, args.negative_label
    )
    y_test = encode_target(
        test_df[args.target_col], args.positive_label, args.negative_label
    )

    print(
        f"Train={len(train_df)} | Test={len(test_df)} | "
        f"Positive rate train={y_train.mean():.3f} | test={y_test.mean():.3f}"
    )

    results = {}

    metrics, pred_df, model = run_dummy_baseline(
        y_train, y_test, args.positive_label, args.negative_label
    )
    results["majority_baseline"] = metrics
    pred_out = test_df[id_cols].copy() if id_cols else pd.DataFrame(index=test_df.index)
    pred_out = pd.concat([pred_out.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
    pred_out.to_parquet(Path(args.output_dir) / "predictions_majority_baseline.parquet", index=False)
    joblib.dump(model, Path(args.output_dir) / "model_majority_baseline.joblib")

    experiments = [
        (
            "jungle_only_logreg",
            build_logreg_pipeline(
                feature_sets["jungle_only"],
                class_weight_balanced=args.class_weight_balanced,
            ),
            feature_sets["jungle_only"],
        ),
        (
            "jungle_mid_support_logreg",
            build_logreg_pipeline(
                feature_sets["jungle_mid_support"],
                class_weight_balanced=args.class_weight_balanced,
            ),
            feature_sets["jungle_mid_support"],
        ),
        (
            "full_draft_logreg",
            build_logreg_pipeline(
                feature_sets["full_draft"],
                class_weight_balanced=args.class_weight_balanced,
            ),
            feature_sets["full_draft"],
        ),
        (
            "random_forest_jungle_only",
            build_random_forest_pipeline(
                feature_sets["jungle_only"],
                n_estimators=args.rf_n_estimators,
                max_depth=args.rf_max_depth,
                min_samples_leaf=args.rf_min_samples_leaf,
                random_state=args.random_state,
                class_weight_balanced=args.class_weight_balanced,
            ),
            feature_sets["jungle_only"],
        ),
        (
            "random_forest_full_draft",
            build_random_forest_pipeline(
                feature_sets["full_draft"],
                n_estimators=args.rf_n_estimators,
                max_depth=args.rf_max_depth,
                min_samples_leaf=args.rf_min_samples_leaf,
                random_state=args.random_state,
                class_weight_balanced=args.class_weight_balanced,
            ),
            feature_sets["full_draft"],
        ),
    ]

    for model_name, pipeline, feature_cols in experiments:
        metrics, pred_df, model = run_model(
            model_name=model_name,
            pipeline=pipeline,
            feature_cols=feature_cols,
            train_df=train_df,
            y_train=y_train,
            test_df=test_df,
            y_test=y_test,
            positive_label=args.positive_label,
            negative_label=args.negative_label,
        )
        results[model_name] = metrics
        pred_out = test_df[id_cols].copy() if id_cols else pd.DataFrame(index=test_df.index)
        pred_out = pd.concat([pred_out.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
        pred_out.to_parquet(Path(args.output_dir) / f"predictions_{model_name}.parquet", index=False)
        joblib.dump(model, Path(args.output_dir) / f"model_{model_name}.joblib")

    save_json(results, str(Path(args.output_dir) / "metrics_summary.json"))

    ranking = []
    for name, metrics in results.items():
        ranking.append(
            {
                "model_name": name,
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "f1": metrics.get("f1"),
                "roc_auc": metrics.get("roc_auc"),
                "n_test": metrics.get("n_test"),
            }
        )
    ranking_df = pd.DataFrame(ranking).sort_values(
        by=["balanced_accuracy", "roc_auc", "accuracy"],
        ascending=False,
    )
    ranking_df.to_parquet(Path(args.output_dir) / "model_ranking.parquet", index=False)
    ranking_df.to_csv(Path(args.output_dir) / "model_ranking.csv", index=False)

    print("\n=== Ranking de modelos ===")
    print(ranking_df.to_string(index=False))
    print(f"\nResultados guardados en: {args.output_dir}")


if __name__ == "__main__":
    main()
