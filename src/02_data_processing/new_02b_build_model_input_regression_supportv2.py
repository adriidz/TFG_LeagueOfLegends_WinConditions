#!/usr/bin/env python3
"""
new_02b_build_model_input_regression_supportv2.py

Construye un model_input para regresión multi-output uniendo:
- draft_features.parquet
- jungle_scores.parquet
- support_scores.parquet
- team_tendency_scores.parquet

Versión adaptada para usar support v2 sin romper el trainer actual:
- selecciona una columna fuente de support (por defecto: support_roam_score_v2)
- la renombra a la columna canónica `support_roam_score` en el model_input final
- opcionalmente arrastra columnas auxiliares de support v2 (p. ej. confianza, xp ratio)

Así, el pipeline de entrenamiento puede seguir esperando:
    jungle_presence_score
    support_roam_score
    team_side_focus_score
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

JOIN_KEYS = ["match_id", "team_id"]

JUNGLE_SCORE_COL = "jungle_presence_score"
SUPPORT_SCORE_COL = "support_roam_score"   # nombre canónico en model_input
TEAM_SCORE_COL = "team_side_focus_score"

DEFAULT_DRAFT_PATH = os.path.join("data_new", "clean", "features", "draft_features.parquet")
DEFAULT_JUNGLE_SCORES_PATH = os.path.join("data_new", "clean", "scores", "jungle_scores.parquet")
DEFAULT_SUPPORT_SCORES_PATH = os.path.join("data_new", "clean", "scores", "support_scores.parquet")
DEFAULT_TEAM_SCORES_PATH = os.path.join("data_new", "clean", "scores", "team_tendency_scores.parquet")
DEFAULT_OUT_PATH = os.path.join("data_new", "training", "model_input_multioutput_regression.parquet")
DEFAULT_SUMMARY_DIR = None


# -----------------------------
# Path helpers
# -----------------------------
def format_window_tag(max_minute: float) -> str:
    rounded = int(round(float(max_minute)))
    return f"m{rounded:02d}"


def apply_window_suffix(path: Optional[str], max_minute: Optional[float]) -> Optional[str]:
    if path is None or max_minute is None:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_{format_window_tag(max_minute)}{ext}"


def apply_sample_suffix(path: Optional[str], frac: Optional[float]) -> Optional[str]:
    if path is None or frac is None or frac >= 1.0 or frac <= 0.0:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_sample{int(frac * 100)}{ext}"


def append_suffix_to_dir(path: Optional[str], suffix: Optional[str]) -> Optional[str]:
    if path is None or not suffix:
        return path
    return f"{path}_{suffix}"


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye model_input continuo usando support v2 como target canónico de support."
    )
    parser.add_argument("--draft-path", default=None, help="Ruta base de draft_features.parquet.")
    parser.add_argument("--jungle-scores-path", default=None, help="Ruta base de jungle_scores.parquet.")
    parser.add_argument("--support-scores-path", default=None, help="Ruta base de support_scores.parquet.")
    parser.add_argument("--team-scores-path", default=None, help="Ruta base de team_tendency_scores.parquet.")
    parser.add_argument("--out-path", default=None, help="Ruta base de salida del model_input parquet.")
    parser.add_argument("--summary-dir", default=None, help="Directorio base para tablas de análisis.")
    parser.add_argument(
        "--join-how",
        choices=["left", "inner"],
        default="left",
        help="Tipo de join desde draft hacia score tables. Recomendado: left.",
    )
    parser.add_argument(
        "--allow-missing-scores",
        action="store_true",
        help="Si se activa, no exige score disponible en las 3 tareas.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Fracción (ej 0.1) o lee TFG_SAMPLE_FRAC para ajustar rutas por defecto.",
    )
    parser.add_argument(
        "--score-max-minute",
        type=float,
        default=None,
        help="Carga score tables generadas para una ventana concreta (ej. 11 -> *_m11.parquet).",
    )

    # soporte v2
    parser.add_argument(
        "--support-score-source-col",
        default="support_roam_score_v2",
        help=(
            "Columna fuente a usar como target de support en la tabla support_scores. "
            "Por defecto usa support_roam_score_v2 y la renombra a support_roam_score en el model_input."
        ),
    )
    parser.add_argument(
        "--support-extra-cols",
        nargs="*",
        default=["support_score_confidence_v2", "support_adc_xp_ratio_v2", "mean_distance_to_adc_v2"],
        help="Columnas extra de support a arrastrar si existen.",
    )
    return parser.parse_args()


# -----------------------------
# Generic utils
# -----------------------------
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


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


def load_parquet(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    if p.suffix.lower() != ".parquet":
        raise ValueError(f"Se esperaba un .parquet: {path}")
    return pd.read_parquet(p)


def validate_required_columns(df: pd.DataFrame, name: str, required_cols: Sequence[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Faltan columnas obligatorias en {name}: {missing}")


def validate_no_duplicate_keys(df: pd.DataFrame, name: str) -> None:
    dup_mask = df.duplicated(subset=JOIN_KEYS, keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows > 0:
        dup_preview = df.loc[dup_mask, JOIN_KEYS].head(10)
        raise SystemExit(
            f"Se han encontrado {n_dup_rows} filas duplicadas por {JOIN_KEYS} en {name}.\n"
            f"Primeros duplicados:\n{dup_preview.to_string(index=False)}"
        )


def build_overlap_summary(left_df: pd.DataFrame, right_df: pd.DataFrame, right_name: str) -> pd.DataFrame:
    left_keys = left_df[JOIN_KEYS].drop_duplicates().copy()
    right_keys = right_df[JOIN_KEYS].drop_duplicates().copy()

    left_key_set = set(map(tuple, left_keys.to_numpy()))
    right_key_set = set(map(tuple, right_keys.to_numpy()))

    overlap = left_key_set & right_key_set
    left_only = left_key_set - right_key_set
    right_only = right_key_set - left_key_set

    return pd.DataFrame(
        [{
            "table": right_name,
            "left_rows": len(left_df),
            "right_rows": len(right_df),
            "left_unique_keys": len(left_key_set),
            "right_unique_keys": len(right_key_set),
            "overlap_keys": len(overlap),
            "left_only_keys": len(left_only),
            "right_only_keys": len(right_only),
        }]
    )


def select_score_columns(score_df: pd.DataFrame, score_col: str, include_run_meta: bool, extras: Optional[List[str]] = None) -> pd.DataFrame:
    keep = JOIN_KEYS + [score_col]
    if include_run_meta:
        keep.extend([c for c in ("max_minute", "window_tag") if c in score_df.columns])
    if extras:
        keep.extend([c for c in extras if c in score_df.columns])
    keep = list(dict.fromkeys([c for c in keep if c in score_df.columns]))
    return score_df[keep].copy()


def add_coverage_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["has_jungle_score"] = out[JUNGLE_SCORE_COL].notna() if JUNGLE_SCORE_COL in out.columns else False
    out["has_support_score"] = out[SUPPORT_SCORE_COL].notna() if SUPPORT_SCORE_COL in out.columns else False
    out["has_team_score"] = out[TEAM_SCORE_COL].notna() if TEAM_SCORE_COL in out.columns else False
    out["has_all_three_scores"] = (
        out["has_jungle_score"] & out["has_support_score"] & out["has_team_score"]
    )
    return out


def build_task_coverage_summary(model_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "rows": len(model_df),
        "unique_match_team_keys": int(model_df[JOIN_KEYS].drop_duplicates().shape[0]),
        "has_jungle_score": int(model_df["has_jungle_score"].sum()) if "has_jungle_score" in model_df.columns else 0,
        "has_support_score": int(model_df["has_support_score"].sum()) if "has_support_score" in model_df.columns else 0,
        "has_team_score": int(model_df["has_team_score"].sum()) if "has_team_score" in model_df.columns else 0,
        "has_all_three_scores": int(model_df["has_all_three_scores"].sum()) if "has_all_three_scores" in model_df.columns else 0,
    }])


def build_score_summary(model_df: pd.DataFrame, score_col: str, task: str) -> pd.DataFrame:
    if score_col not in model_df.columns:
        return pd.DataFrame()
    valid = model_df[score_col].dropna()
    if valid.empty:
        return pd.DataFrame()
    return pd.DataFrame([{
        "task": task,
        "score_col": score_col,
        "n": int(valid.shape[0]),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "min": float(valid.min()),
        "q25": float(valid.quantile(0.25)),
        "median": float(valid.quantile(0.50)),
        "q75": float(valid.quantile(0.75)),
        "max": float(valid.max()),
    }])


def save_df(df: pd.DataFrame, path_no_ext: str) -> None:
    ensure_dir(str(Path(path_no_ext).parent))
    df.to_csv(path_no_ext + ".csv", index=False)
    try:
        df.to_parquet(path_no_ext + ".parquet", index=False)
    except Exception:
        pass


def write_summary_tables(
    draft_df: pd.DataFrame,
    task_tables: Dict[str, pd.DataFrame],
    model_df: pd.DataFrame,
    summary_dir: str,
    support_source_col: str,
) -> None:
    ensure_dir(summary_dir)

    overlap_parts: List[pd.DataFrame] = []
    for task_name, task_df in task_tables.items():
        overlap_parts.append(build_overlap_summary(draft_df, task_df, task_name))
    overlap_summary = pd.concat(overlap_parts, ignore_index=True) if overlap_parts else pd.DataFrame()
    if not overlap_summary.empty:
        save_df(overlap_summary, str(Path(summary_dir) / "join_overlap_summary"))

    overall_summary = pd.DataFrame([{
        "model_rows": len(model_df),
        "model_unique_matches": int(model_df["match_id"].nunique()) if "match_id" in model_df.columns else None,
        "model_unique_teams": int(model_df["team_id"].nunique()) if "team_id" in model_df.columns else None,
        "model_unique_match_team_keys": int(model_df[JOIN_KEYS].drop_duplicates().shape[0]),
        "window_tag": model_df["window_tag"].dropna().iloc[0] if "window_tag" in model_df.columns and model_df["window_tag"].notna().any() else None,
        "score_max_minute": float(model_df["max_minute"].dropna().iloc[0]) if "max_minute" in model_df.columns and model_df["max_minute"].notna().any() else None,
        "support_score_source_col": support_source_col,
    }])
    save_df(overall_summary, str(Path(summary_dir) / "overall_model_input_summary"))

    coverage_summary = build_task_coverage_summary(model_df)
    save_df(coverage_summary, str(Path(summary_dir) / "task_coverage_summary"))

    score_summary_parts = [
        build_score_summary(model_df, JUNGLE_SCORE_COL, "jungle"),
        build_score_summary(model_df, SUPPORT_SCORE_COL, "support"),
        build_score_summary(model_df, TEAM_SCORE_COL, "team"),
    ]
    non_empty = [df for df in score_summary_parts if not df.empty]
    if non_empty:
        score_summary = pd.concat(non_empty, ignore_index=True)
        save_df(score_summary, str(Path(summary_dir) / "score_summary_by_task"))

    if "patch" in model_df.columns:
        patch_counts = (
            model_df.groupby("patch", dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
        )
        save_df(patch_counts, str(Path(summary_dir) / "patch_counts"))

    if "side" in model_df.columns:
        side_counts = (
            model_df.groupby("side", dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
        )
        save_df(side_counts, str(Path(summary_dir) / "side_counts"))


# -----------------------------
# Path resolution
# -----------------------------
def resolve_base_paths(args: argparse.Namespace) -> Tuple[str, str, str, str, str, Optional[str]]:
    target_frac = get_target_frac(args.sample_frac)

    draft_path = args.draft_path or DEFAULT_DRAFT_PATH
    jungle_scores_path = args.jungle_scores_path or DEFAULT_JUNGLE_SCORES_PATH
    support_scores_path = args.support_scores_path or DEFAULT_SUPPORT_SCORES_PATH
    team_scores_path = args.team_scores_path or DEFAULT_TEAM_SCORES_PATH
    out_path = args.out_path or DEFAULT_OUT_PATH
    summary_dir = args.summary_dir or DEFAULT_SUMMARY_DIR

    if target_frac is not None and 0.0 < target_frac < 1.0:
        draft_path = apply_sample_suffix(draft_path, target_frac)
        jungle_scores_path = apply_sample_suffix(jungle_scores_path, target_frac)
        support_scores_path = apply_sample_suffix(support_scores_path, target_frac)
        team_scores_path = apply_sample_suffix(team_scores_path, target_frac)
        out_path = apply_sample_suffix(out_path, target_frac)
        print(f"Muestreo detectado ({target_frac}). Rutas ajustadas automáticamente a sufijos _sample.")

    if args.score_max_minute is not None:
        jungle_scores_path = apply_window_suffix(jungle_scores_path, args.score_max_minute)
        support_scores_path = apply_window_suffix(support_scores_path, args.score_max_minute)
        team_scores_path = apply_window_suffix(team_scores_path, args.score_max_minute)
        out_path = apply_window_suffix(out_path, args.score_max_minute)
        if summary_dir:
            summary_dir = append_suffix_to_dir(summary_dir, format_window_tag(args.score_max_minute))
        print(f"Cargando score tables para ventana 0-{args.score_max_minute:g} min ({format_window_tag(args.score_max_minute)}).")

    return draft_path, jungle_scores_path, support_scores_path, team_scores_path, out_path, summary_dir


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()

    draft_path, jungle_scores_path, support_scores_path, team_scores_path, out_path_str, summary_dir = resolve_base_paths(args)

    out_path = Path(out_path_str)
    ensure_dir(str(out_path.parent))
    summary_dir = summary_dir or (str(out_path.with_suffix("")) + "_analysis")

    print("\n[Rutas] Cargando draft features (Entrada):", os.path.abspath(draft_path))
    draft_df = load_parquet(draft_path)
    validate_required_columns(draft_df, "draft_features", JOIN_KEYS)
    validate_no_duplicate_keys(draft_df, "draft_features")

    model_df = draft_df.copy()
    loaded_task_tables: Dict[str, pd.DataFrame] = {}

    # Jungle
    print(f"[Rutas] Cargando jungle scores (Entrada): {os.path.abspath(jungle_scores_path)}")
    jungle_df = load_parquet(jungle_scores_path)
    validate_required_columns(jungle_df, "jungle_scores", JOIN_KEYS + [JUNGLE_SCORE_COL])
    validate_no_duplicate_keys(jungle_df, "jungle_scores")
    jungle_df = select_score_columns(jungle_df, JUNGLE_SCORE_COL, include_run_meta=True)
    loaded_task_tables["jungle"] = jungle_df.copy()
    model_df = model_df.merge(jungle_df, on=JOIN_KEYS, how=args.join_how, validate="one_to_one")
    print(f"  - Filas tras unir jungle: {len(model_df)}")

    # Support (columna configurable -> renombrada a nombre canónico)
    print(f"[Rutas] Cargando support scores (Entrada): {os.path.abspath(support_scores_path)}")
    support_df = load_parquet(support_scores_path)
    validate_required_columns(support_df, "support_scores", JOIN_KEYS + [args.support_score_source_col])
    validate_no_duplicate_keys(support_df, "support_scores")
    support_df = select_score_columns(
        support_df,
        args.support_score_source_col,
        include_run_meta=False,
        extras=args.support_extra_cols,
    )
    if args.support_score_source_col != SUPPORT_SCORE_COL:
        support_df = support_df.rename(columns={args.support_score_source_col: SUPPORT_SCORE_COL})
    loaded_task_tables["support"] = support_df.copy()
    model_df = model_df.merge(support_df, on=JOIN_KEYS, how=args.join_how, validate="one_to_one")
    print(f"  - Filas tras unir support: {len(model_df)}")

    # Team
    print(f"[Rutas] Cargando team scores (Entrada): {os.path.abspath(team_scores_path)}")
    team_df = load_parquet(team_scores_path)
    validate_required_columns(team_df, "team_scores", JOIN_KEYS + [TEAM_SCORE_COL])
    validate_no_duplicate_keys(team_df, "team_scores")
    team_df = select_score_columns(team_df, TEAM_SCORE_COL, include_run_meta=False)
    loaded_task_tables["team"] = team_df.copy()
    model_df = model_df.merge(team_df, on=JOIN_KEYS, how=args.join_how, validate="one_to_one")
    print(f"  - Filas tras unir team: {len(model_df)}")

    model_df = add_coverage_columns(model_df)

    if not args.allow_missing_scores:
        before = len(model_df)
        model_df = model_df[model_df["has_all_three_scores"]].copy()
        print(f"Filas tras exigir has_all_three_scores=True: {len(model_df)} (eliminadas {before - len(model_df)})")

    validate_no_duplicate_keys(model_df, "model_input_multioutput_regression")

    print(f"\n[Rutas] model_input parquet se guardará en (Salida): {os.path.abspath(out_path)}")
    print(f"[Rutas] Reportes de unión y análisis se guardarán en (Salida): {os.path.abspath(summary_dir)}\n")

    print(f"Draft rows: {len(draft_df)}")
    for task_name, task_df in loaded_task_tables.items():
        print(f"{task_name.title()} score rows: {len(task_df)}")
    print(f"Filas finales model_input: {len(model_df)}")

    print("\n[DEBUG] Targets continuos presentes:")
    for col in (JUNGLE_SCORE_COL, SUPPORT_SCORE_COL, TEAM_SCORE_COL):
        print(f"[DEBUG] {col}: {'OK' if col in model_df.columns else 'MISSING'}")
    print(f"[DEBUG] support source used: {args.support_score_source_col} -> {SUPPORT_SCORE_COL}")

    model_df.to_parquet(out_path, index=False)
    write_summary_tables(draft_df, loaded_task_tables, model_df, summary_dir, args.support_score_source_col)

    print("\nHecho.")
    print(f"- model input parquet: {out_path}")
    print(f"- analysis dir: {summary_dir}")


if __name__ == "__main__":
    main()
