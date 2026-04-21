#!/usr/bin/env python3
"""
new_03_train_multioutput_regression.py

Entrena un modelo multi-output de REGRESIÓN usando:
- entradas categóricas del draft
- One-Hot Encoding
- un modelo lineal multi-output (LinearRegression por defecto)

Objetivo:
    predecir directamente los scores continuos crudos:
        jungle_presence_score
        support_roam_score
        team_side_focus_score

Esta rama es deliberadamente más simple que la de clasificación con embeddings:
1) no discretiza targets antes de entrenar,
2) conserva información de distancia,
3) la salida tiene exactamente 3 valores continuos.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_INPUT_PATH = os.path.join("data_new", "training", "model_input_multioutput_regression.parquet")
DEFAULT_OUTPUT_DIR = os.path.join("Models_new", "multioutput_regression_linear")
DEFAULT_GROUP_COL = "match_id"

TARGET_COLUMNS = [
    "jungle_presence_score",
    "support_roam_score",
    "team_side_focus_score",
]

ROLE_KEYS = ("top", "jungle", "middle", "bottom", "utility")
SIDES = ("ally", "enemy")

FEATURE_GROUP_DEFS: Dict[str, Dict[str, Any]] = {
    "champions": {
        "columns": [f"{s}_{r}_champion_id" for s in SIDES for r in ROLE_KEYS],
        "description": "Champion picks de ambos equipos (10 posiciones).",
    },
    "summoner_spells": {
        "columns": [f"{s}_{r}_summoner{i}_id" for s in SIDES for r in ROLE_KEYS for i in (1, 2)],
        "description": "Hechizos de invocador (Flash, Teleport, Ignite...).",
    },
    "keystones": {
        "columns": [f"{s}_{r}_keystone_id" for s in SIDES for r in ROLE_KEYS],
        "description": "Runa keystone por jugador.",
    },
    "rune_styles": {
        "columns": [
            f"{s}_{r}_{style}_style_id"
            for s in SIDES for r in ROLE_KEYS for style in ("primary", "sub")
        ],
        "description": "Árboles de runas primario y secundario.",
    },
    "bans": {
        "columns": [f"{s}_ban_{i}_champion_id" for s in SIDES for i in range(1, 6)],
        "description": "Bans de ambos equipos.",
    },
    "context": {
        "columns": ["side"],
        "description": "Side (blue/red).",
    },
}

ALL_GROUP_NAMES = list(FEATURE_GROUP_DEFS.keys())
ABLATION_PRESETS = {
    "minimal": ["champions", "context"],
    "standard": ["champions", "summoner_spells", "context"],
    "draft_plus_runes": ["champions", "summoner_spells", "keystones", "rune_styles", "context"],
    "all": ALL_GROUP_NAMES,
}


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def format_window_tag(max_minute: float) -> str:
    rounded = int(round(float(max_minute)))
    return f"m{rounded:02d}"


def apply_window_suffix(path: str, max_minute: Optional[float]) -> str:
    if max_minute is None:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_{format_window_tag(max_minute)}{ext}"


def apply_sample_suffix(path: str, frac: Optional[float]) -> str:
    if frac is None or frac >= 1.0 or frac <= 0.0:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_sample{int(frac * 100)}{ext}"


def get_target_frac(args_frac: Optional[float]) -> Optional[float]:
    if args_frac is not None:
        return args_frac
    env_frac = os.getenv("TFG_SAMPLE_FRAC")
    if env_frac:
        try:
            return float(env_frac)
        except ValueError:
            pass
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regresión multi-output simple con One-Hot + modelo lineal.")
    p.add_argument("--input", default=DEFAULT_INPUT_PATH)
    p.add_argument("--outdir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--score-max-minute", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument(
        "--feature-groups", nargs="*", default=None,
        help=f"Grupos de features a usar. Opciones: {ALL_GROUP_NAMES}. Presets: {list(ABLATION_PRESETS.keys())}."
    )
    p.add_argument(
        "--regressor",
        choices=["linear", "ridge"],
        default="linear",
        help="Modelo final sobre la matriz One-Hot. 'linear' = OLS puro; 'ridge' = MSE + regularización L2.",
    )
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Solo se usa si --regressor ridge.")
    return p.parse_args()


def validate_targets(df: pd.DataFrame, target_cols: List[str]) -> None:
    print("\n[Validación] Comprobando targets continuos...")
    for col in target_cols:
        if col not in df.columns:
            raise SystemExit(f"Falta target obligatorio: {col}")
        valid = df[col].dropna()
        if valid.empty:
            raise SystemExit(f"Target sin valores válidos: {col}")
        print(
            f"  ✓ {col}: n={len(valid)} | mean={valid.mean():.4f} | std={valid.std():.4f} "
            f"| min={valid.min():.4f} | max={valid.max():.4f}"
        )


def select_active_groups(feature_groups_arg: Optional[List[str]]) -> List[str]:
    if feature_groups_arg is None:
        return list(ALL_GROUP_NAMES)
    if len(feature_groups_arg) == 1 and feature_groups_arg[0] in ABLATION_PRESETS:
        return ABLATION_PRESETS[feature_groups_arg[0]]
    active = [g for g in feature_groups_arg if g in FEATURE_GROUP_DEFS]
    if not active:
        raise SystemExit(f"No hay feature groups activos. Opciones: {ALL_GROUP_NAMES}")
    return active


def resolve_paths(args: argparse.Namespace) -> Tuple[str, str]:
    target_frac = get_target_frac(args.sample_frac)
    in_path = args.input
    out_dir = args.outdir

    if target_frac is not None and 0.0 < target_frac < 1.0:
        in_path = apply_sample_suffix(in_path, target_frac)
        out_dir = apply_sample_suffix(out_dir, target_frac)
    if args.score_max_minute is not None:
        in_path = apply_window_suffix(in_path, args.score_max_minute)
        out_dir = apply_window_suffix(out_dir, args.score_max_minute)

    return in_path, out_dir


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    for idx, target_name in enumerate(target_names):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        mse = float(mean_squared_error(yt, yp))
        rmse = math.sqrt(mse)
        mae = float(mean_absolute_error(yt, yp))
        r2 = float(r2_score(yt, yp))
        corr = float(np.corrcoef(yt, yp)[0, 1]) if np.std(yt) > 0 and np.std(yp) > 0 else float("nan")
        rows.append({
            "target": target_name,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "pearson_corr": corr,
        })
    metrics_df = pd.DataFrame(rows)
    overall = {
        "mean_mse": float(metrics_df["mse"].mean()),
        "mean_rmse": float(metrics_df["rmse"].mean()),
        "mean_mae": float(metrics_df["mae"].mean()),
        "mean_r2": float(metrics_df["r2"].mean()),
        "mean_pearson_corr": float(metrics_df["pearson_corr"].mean()),
    }
    return metrics_df, overall


def build_architecture_markdown(
    raw_feature_columns: List[str],
    active_groups: List[str],
    onehot_dim: int,
    regressor_name: str,
    target_names: List[str],
) -> str:
    lines = []
    lines.append("# Arquitectura del modelo de regresión\n")
    lines.append("## Unidad de muestra")
    lines.append("Cada fila representa un equipo dentro de una partida: `(match_id, team_id)`.\n")
    lines.append("## Entrada cruda")
    lines.append(f"- Columnas de entrada antes de One-Hot: **{len(raw_feature_columns)}**")
    lines.append(f"- Grupos activos: **{', '.join(active_groups)}**")
    lines.append("")
    for g in active_groups:
        lines.append(f"- **{g}**: {FEATURE_GROUP_DEFS[g]['description']}")
    lines.append("\n## Preprocesado")
    lines.append("- Todas las columnas categóricas seleccionadas se transforman con `OneHotEncoder(handle_unknown='ignore')`.")
    lines.append(f"- Dimensión final tras One-Hot en train: **{onehot_dim}**.\n")
    lines.append("## Modelo")
    if regressor_name == "linear":
        lines.append("- Modelo final: **LinearRegression multi-output**.")
        lines.append("- Ecuación conceptual: `ŷ = XW + b`.")
        lines.append("- Esto equivale a una única capa lineal con **3 salidas continuas**.")
    else:
        lines.append("- Modelo final: **Ridge multi-output**.")
        lines.append("- Ecuación conceptual: `ŷ = XW + b`.")
        lines.append("- Optimiza error cuadrático con regularización L2.")
        lines.append("- También equivale a una única capa lineal con **3 salidas continuas**.")
    lines.append("\n## Salida")
    lines.append(f"- Número de salidas: **{len(target_names)}**")
    for t in target_names:
        lines.append(f"- `{t}`")
    lines.append("\n## Por qué esta versión es más simple")
    lines.append("- No discretiza los scores antes de entrenar.")
    lines.append("- Conserva la magnitud del error.")
    lines.append("- La arquitectura completa es fácil de explicar: **One-Hot + capa lineal de 3 salidas**.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    active_groups = select_active_groups(args.feature_groups)
    in_path, out_dir = resolve_paths(args)

    print(f"\n{'='*70}")
    print(" Regresión Multi-Output Simple")
    print(f" Regresor: {args.regressor}")
    print(f" Feature groups: {active_groups}")
    if args.score_max_minute is not None:
        print(f" Score window: {format_window_tag(args.score_max_minute)}")
    print(f"{'='*70}")

    print(f"\n[Datos] Cargando: {os.path.abspath(in_path)}")
    if not os.path.exists(in_path):
        raise SystemExit(f"Archivo no encontrado: {in_path}")
    df = pd.read_parquet(in_path)
    print(f"        Filas: {len(df)}")

    validate_targets(df, TARGET_COLUMNS)

    before = len(df)
    df = df.dropna(subset=TARGET_COLUMNS).copy()
    print(f"\n[Filtro] Filas con los 3 targets presentes: {len(df)} (eliminadas {before - len(df)})")

    feature_columns: List[str] = []
    group_sizes: List[Dict[str, Any]] = []
    for gname in active_groups:
        cols_present = [c for c in FEATURE_GROUP_DEFS[gname]["columns"] if c in df.columns]
        feature_columns.extend(cols_present)
        group_sizes.append({"group": gname, "n_columns_present": len(cols_present)})

    if not feature_columns:
        raise SystemExit("No hay features disponibles en el parquet para los grupos solicitados.")

    print(f"\n[Features] {len(feature_columns)} columnas crudas antes de One-Hot")
    for row in group_sizes:
        print(f"  - {row['group']}: {row['n_columns_present']} columnas")

    print(f"\n[Split] GroupShuffleSplit por {DEFAULT_GROUP_COL}...")
    groups = df[DEFAULT_GROUP_COL]
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(df, groups=groups))
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()
    print(f"        Train: {len(df_train)} | Val: {len(df_val)}")

    X_train = df_train[feature_columns].copy()
    X_val = df_val[feature_columns].copy()
    y_train = df_train[TARGET_COLUMNS].to_numpy(dtype=float)
    y_val = df_val[TARGET_COLUMNS].to_numpy(dtype=float)

    ohe = OneHotEncoder(handle_unknown="ignore")
    if args.regressor == "linear":
        regressor = LinearRegression()
    else:
        regressor = Ridge(alpha=args.alpha)

    pipeline = Pipeline(steps=[
        ("onehot", ohe),
        ("regressor", regressor),
    ])

    print("\n[Entrenamiento] Ajustando One-Hot + modelo lineal...")
    pipeline.fit(X_train, y_train)

    fitted_ohe: OneHotEncoder = pipeline.named_steps["onehot"]
    X_train_ohe = fitted_ohe.transform(X_train)
    X_val_ohe = fitted_ohe.transform(X_val)
    input_dim = int(X_train_ohe.shape[1])

    print(f"  - Dimensión tras One-Hot (train): {input_dim}")
    print(f"  - Dimensión de salida: {len(TARGET_COLUMNS)}")

    y_pred = pipeline.predict(X_val)
    metrics_df, overall_metrics = compute_metrics(y_val, y_pred, TARGET_COLUMNS)

    print("\n[Métricas de validación]")
    print(metrics_df.to_string(index=False))
    print("\n[Promedio]")
    for k, v in overall_metrics.items():
        print(f"  - {k}: {v:.6f}")

    ensure_dir(out_dir)
    model_path = os.path.join(out_dir, "regression_pipeline.joblib")
    joblib.dump(pipeline, model_path)

    metrics_path = os.path.join(out_dir, "metrics_by_target.csv")
    metrics_df.to_csv(metrics_path, index=False)

    predictions_df = df_val[["match_id", "team_id"] + ([c for c in ("side", "patch") if c in df_val.columns])].copy()
    for idx, target in enumerate(TARGET_COLUMNS):
        predictions_df[f"true_{target}"] = y_val[:, idx]
        predictions_df[f"pred_{target}"] = y_pred[:, idx]
        predictions_df[f"abs_error_{target}"] = np.abs(y_val[:, idx] - y_pred[:, idx])
    predictions_path = os.path.join(out_dir, "validation_predictions.parquet")
    predictions_df.to_parquet(predictions_path, index=False)

    feature_space_summary = pd.DataFrame(group_sizes)
    feature_space_summary["raw_feature_columns_total"] = len(feature_columns)
    feature_space_summary["onehot_dimension_train"] = input_dim
    feature_space_summary.to_csv(os.path.join(out_dir, "feature_space_summary.csv"), index=False)

    config = {
        "model_family": "onehot_linear_regression" if args.regressor == "linear" else "onehot_ridge_regression",
        "regressor": args.regressor,
        "alpha": args.alpha if args.regressor == "ridge" else None,
        "target_columns": TARGET_COLUMNS,
        "feature_groups_active": active_groups,
        "feature_groups_available": ALL_GROUP_NAMES,
        "feature_columns": feature_columns,
        "raw_input_dimension": len(feature_columns),
        "onehot_input_dimension_train": input_dim,
        "output_dimension": len(TARGET_COLUMNS),
        "sample_frac": get_target_frac(args.sample_frac),
        "score_max_minute": args.score_max_minute,
        "score_window_tag": format_window_tag(args.score_max_minute) if args.score_max_minute is not None else None,
        "val_size": args.val_size,
        "seed": args.seed,
        "n_train": len(df_train),
        "n_val": len(df_val),
        "resolved_input_path": os.path.abspath(in_path),
        "resolved_output_dir": os.path.abspath(out_dir),
        "metrics_by_target": metrics_df.to_dict(orient="records"),
        "overall_metrics": overall_metrics,
    }
    with open(os.path.join(out_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    architecture_md = build_architecture_markdown(
        raw_feature_columns=feature_columns,
        active_groups=active_groups,
        onehot_dim=input_dim,
        regressor_name=args.regressor,
        target_names=TARGET_COLUMNS,
    )
    with open(os.path.join(out_dir, "architecture_summary.md"), "w", encoding="utf-8") as f:
        f.write(architecture_md)

    print(f"\n[Exportación] Guardando en: {os.path.abspath(out_dir)}")
    print("  ✓ regression_pipeline.joblib")
    print("  ✓ metrics_by_target.csv")
    print("  ✓ validation_predictions.parquet")
    print("  ✓ feature_space_summary.csv")
    print("  ✓ model_config.json")
    print("  ✓ architecture_summary.md")
    print("\nHecho.")


if __name__ == "__main__":
    main()
