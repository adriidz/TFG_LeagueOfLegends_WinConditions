
#!/usr/bin/env python3
"""
train_jungle_presence_model.py

Entrena un primer modelo para predecir la etiqueta de jungla
("map_presence" vs "farm_oriented") a partir de variables de draft.

Entrada esperada:
- Un parquet tabular con una fila por (match_id, team_id)
- Debe incluir una columna objetivo: jungle_presence_label
- Debe incluir columnas categóricas de draft como, por ejemplo:
    ally_top_champion_name
    ally_jungle_champion_name
    ally_middle_champion_name
    ally_bottom_champion_name
    ally_utility_champion_name
    enemy_top_champion_name
    enemy_jungle_champion_name
    enemy_middle_champion_name
    enemy_bottom_champion_name
    enemy_utility_champion_name
    side
    patch
  (o un subconjunto compatible)

El script:
- filtra la clase "ambiguous"
- compara 3 modelos:
    1) baseline mayoritario
    2) baseline solo campeón jungla
    3) draft completo
- usa OneHotEncoder + LogisticRegression
- guarda métricas, predicciones y el mejor pipeline
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un primer modelo de jungla presence desde draft."
    )
    parser.add_argument(
        "--input-path",
        default=os.path.join("Data_training", "model_input_jungle.parquet"),
        help="Parquet de entrada con features + target.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("Models", "jungle_presence_logreg"),
        help="Directorio donde guardar métricas, predicciones y modelo.",
    )
    parser.add_argument(
        "--target-col",
        default=TARGET_COL,
        help=f"Nombre de la columna objetivo (default: {TARGET_COL}).",
    )
    parser.add_argument(
        "--positive-label",
        default=POSITIVE_LABEL,
        help=f"Etiqueta positiva (default: {POSITIVE_LABEL}).",
    )
    parser.add_argument(
        "--negative-label",
        default=NEGATIVE_LABEL,
        help=f"Etiqueta negativa (default: {NEGATIVE_LABEL}).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporción de test (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla aleatoria (default: 42).",
    )
    parser.add_argument(
        "--full-features",
        nargs="*",
        default=DEFAULT_FULL_FEATURES,
        help="Columnas a usar en el modelo de draft completo.",
    )
    parser.add_argument(
        "--jungle-only-features",
        nargs="*",
        default=DEFAULT_JUNGLE_ONLY_FEATURES,
        help="Columnas a usar en el baseline de jungle-only.",
    )
    parser.add_argument(
        "--drop-cols",
        nargs="*",
        default=["match_id", "team_id", "jungle_presence_score"],
        help="Columnas que se ignorarán si existen.",
    )
    parser.add_argument(
        "--class-weight-balanced",
        action="store_true",
        help="Usar class_weight='balanced' en LogisticRegression.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo de entrada: {path}")
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError("Solo se admite .parquet como input en esta versión.")


def filter_target_classes(
    df: pd.DataFrame,
    target_col: str,
    positive_label: str,
    negative_label: str,
) -> pd.DataFrame:
    keep = {positive_label, negative_label}
    out = df[df[target_col].isin(keep)].copy()
    if out.empty:
        raise ValueError(
            f"No quedan filas tras filtrar {target_col} por {sorted(keep)}."
        )
    return out


def select_existing_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    return [c for c in columns if c in df.columns]


def encode_target(
    y: pd.Series,
    positive_label: str,
    negative_label: str,
) -> np.ndarray:
    mapping = {negative_label: 0, positive_label: 1}
    return y.map(mapping).astype(int).to_numpy()


def build_logreg_pipeline(
    categorical_features: List[str],
    class_weight_balanced: bool = False,
) -> Pipeline:
    preprocessor = ColumnTransformer(
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

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced" if class_weight_balanced else None,
        solver="lbfgs",
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clf),
        ]
    )
    return pipe


def evaluate_binary_model(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    positive_label: str,
    negative_label: str,
) -> Tuple[Dict, pd.DataFrame]:
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        proba = None

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

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[negative_label, positive_label],
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report"] = report

    pred_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
        }
    )
    if proba is not None:
        pred_df["p_map_presence"] = proba

    return metrics, pred_df


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_dummy_baseline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    positive_label: str,
    negative_label: str,
) -> Tuple[Dict, pd.DataFrame, object]:
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_train, y_train)
    metrics, pred_df = evaluate_binary_model(
        clf, X_test, y_test, positive_label, negative_label
    )
    return metrics, pred_df, clf


def run_logreg_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    categorical_features: List[str],
    positive_label: str,
    negative_label: str,
    class_weight_balanced: bool,
) -> Tuple[Dict, pd.DataFrame, Pipeline]:
    pipe = build_logreg_pipeline(
        categorical_features=categorical_features,
        class_weight_balanced=class_weight_balanced,
    )
    pipe.fit(X_train[categorical_features], y_train)
    metrics, pred_df = evaluate_binary_model(
        pipe,
        X_test[categorical_features],
        y_test,
        positive_label,
        negative_label,
    )
    return metrics, pred_df, pipe


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

    for col in args.drop_cols:
        if col in df.columns:
            pass  # se ignoran después si no están en las feature lists

    # Guardamos una copia ligera con ids si existen, para predicciones.
    id_cols = [c for c in ["match_id", "team_id"] if c in df.columns]

    full_features = select_existing_columns(df, args.full_features)
    jungle_only_features = select_existing_columns(df, args.jungle_only_features)

    if not full_features:
        raise ValueError(
            "No se encontró ninguna columna del conjunto --full-features en el parquet."
        )
    if not jungle_only_features:
        raise ValueError(
            "No se encontró ninguna columna del conjunto --jungle-only-features en el parquet."
        )

    print("Features draft completo:", full_features)
    print("Features jungle-only:", jungle_only_features)

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
        train_df[args.target_col],
        positive_label=args.positive_label,
        negative_label=args.negative_label,
    )
    y_test = encode_target(
        test_df[args.target_col],
        positive_label=args.positive_label,
        negative_label=args.negative_label,
    )

    print(
        f"Train={len(train_df)} | Test={len(test_df)} | "
        f"Positive rate train={y_train.mean():.3f} | test={y_test.mean():.3f}"
    )

    results = {}

    print("Entrenando baseline mayoritario...")
    # Solo necesitamos un dataframe cualquiera con el número correcto de filas.
    dummy_X_train = pd.DataFrame({"dummy": np.zeros(len(train_df), dtype=int)})
    dummy_X_test = pd.DataFrame({"dummy": np.zeros(len(test_df), dtype=int)})
    metrics, pred_df, model = run_dummy_baseline(
        dummy_X_train,
        y_train,
        dummy_X_test,
        y_test,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
    )
    results["majority_baseline"] = metrics
    pred_out = test_df[id_cols].copy() if id_cols else pd.DataFrame(index=test_df.index)
    pred_out = pd.concat([pred_out.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
    pred_out.to_parquet(Path(args.output_dir) / "predictions_majority_baseline.parquet", index=False)
    joblib.dump(model, Path(args.output_dir) / "model_majority_baseline.joblib")

    print("Entrenando baseline jungle-only...")
    metrics, pred_df, model = run_logreg_model(
        train_df,
        y_train,
        test_df,
        y_test,
        categorical_features=jungle_only_features,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
        class_weight_balanced=args.class_weight_balanced,
    )
    results["jungle_only_logreg"] = metrics
    pred_out = test_df[id_cols].copy() if id_cols else pd.DataFrame(index=test_df.index)
    pred_out = pd.concat([pred_out.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
    pred_out.to_parquet(Path(args.output_dir) / "predictions_jungle_only_logreg.parquet", index=False)
    joblib.dump(model, Path(args.output_dir) / "model_jungle_only_logreg.joblib")

    print("Entrenando modelo draft completo...")
    metrics, pred_df, model = run_logreg_model(
        train_df,
        y_train,
        test_df,
        y_test,
        categorical_features=full_features,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
        class_weight_balanced=args.class_weight_balanced,
    )
    results["full_draft_logreg"] = metrics
    pred_out = test_df[id_cols].copy() if id_cols else pd.DataFrame(index=test_df.index)
    pred_out = pd.concat([pred_out.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
    pred_out.to_parquet(Path(args.output_dir) / "predictions_full_draft_logreg.parquet", index=False)
    joblib.dump(model, Path(args.output_dir) / "model_full_draft_logreg.joblib")

    print("Guardando métricas...")
    save_json(results, str(Path(args.output_dir) / "metrics_summary.json"))

    # Ranking rápido por balanced_accuracy.
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
