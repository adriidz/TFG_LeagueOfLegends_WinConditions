#!/usr/bin/env python3
"""
02b_build_model_input.py

Une:
- draft_features.parquet
- jungle_labels.parquet (opcional)
- support_labels.parquet (opcional)
- team_tendency_labels.parquet (opcional)

por:
- match_id
- team_id

y genera un parquet final listo para entrenamiento de modelos single-task o
multi-output.

Filosofía:
- El draft es la tabla base.
- Las tablas de labels se unen por (match_id, team_id).
- Se validan duplicados y columnas requeridas.
- Se añaden indicadores de cobertura por tarea.
- Se pueden exigir las tres tareas presentes y/o eliminar ambiguas.
- Fase 1: permite elegir qué ventana temporal de labels cargar (m06, m08, m10...).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

JOIN_KEYS = ["match_id", "team_id"]

JUNGLE_REQUIRED_ANY = ["jungle_presence_score"]
SUPPORT_REQUIRED_ANY = ["support_roam_score"]
TEAM_REQUIRED_ANY = ["team_side_focus_score"]

JUNGLE_LABEL_COL = "jungle_presence_label"
SUPPORT_LABEL_COL = "support_roam_label"
TEAM_LABEL_COL = "team_tendency_label"

AMBIGUOUS_BY_TASK = {
    "jungle": (JUNGLE_LABEL_COL, "ambiguous"),
    "support": (SUPPORT_LABEL_COL, "ambiguous"),
    "team": (TEAM_LABEL_COL, "ambiguous"),
}

# ── RUTAS (editar aquí para cambiar entrada/salida) ──────────────────────────
DEFAULT_DRAFT_PATH = os.path.join("data", "clean", "features", "draft_features.parquet")
DEFAULT_JUNGLE_LABELS_PATH = os.path.join("data", "clean", "labels", "jungle_labels.parquet")
DEFAULT_SUPPORT_LABELS_PATH = os.path.join("data", "clean", "labels", "support_labels.parquet")
DEFAULT_TEAM_LABELS_PATH = os.path.join("data", "clean", "labels", "team_tendency_labels.parquet")
DEFAULT_OUT_PATH = os.path.join("data", "training", "model_input_multioutput.parquet")
DEFAULT_SUMMARY_DIR = None  # None → se deriva automáticamente de DEFAULT_OUT_PATH


def format_window_tag(max_minute: float) -> str:
    rounded = int(round(float(max_minute)))
    return f"m{rounded:02d}"


def apply_window_suffix(path: Optional[str], max_minute: Optional[float]) -> Optional[str]:
    if path is None or max_minute is None:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_{format_window_tag(max_minute)}{ext}"


def apply_schema_suffix(path: Optional[str], label_schema: Optional[str]) -> Optional[str]:
    if path is None or not label_schema:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_{label_schema}{ext}"



def _fmt_num_for_tag(value: float) -> str:
    if abs(float(value) - round(float(value))) < 1e-9:
        return str(int(round(float(value))))
    return str(value).replace(".", "p")


def format_quantile_or_threshold_tag(
    lower_q: Optional[float],
    upper_q: Optional[float],
    lower_thr: Optional[float],
    upper_thr: Optional[float],
) -> str:
    if lower_thr is not None and upper_thr is not None:
        return f"thr{_fmt_num_for_tag(lower_thr)}_{_fmt_num_for_tag(upper_thr)}"
    return f"q{int(round(float(lower_q) * 100)):02d}_{int(round(float(upper_q) * 100)):02d}"


def apply_quantile_suffix(
    path: Optional[str],
    lower_q: Optional[float],
    upper_q: Optional[float],
    lower_thr: Optional[float],
    upper_thr: Optional[float],
) -> Optional[str]:
    if path is None:
        return path
    base, ext = os.path.splitext(path)
    tag = format_quantile_or_threshold_tag(lower_q, upper_q, lower_thr, upper_thr)
    return f"{base}_{tag}{ext}"

# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye model_input multi-output uniendo draft features con hasta 3 tablas de labels."
    )
    parser.add_argument(
        "--join-how",
        choices=["left", "inner"],
        default="left",
        help="Tipo de join desde draft hacia labels. Recomendado: left.",
    )
    parser.add_argument(
        "--require-all-three-scores",
        action="store_true",
        help="Si se activa, conserva solo filas con score continuo disponible en las 3 tareas.",
    )
    parser.add_argument(
        "--require-all-three-labels",
        action="store_true",
        help="Si se activa, conserva solo filas con label discreta disponible en las 3 tareas.",
    )
    parser.add_argument(
        "--drop-ambiguous-labels",
        action="store_true",
        help="Elimina filas donde alguna label discreta presente sea ambiguous.",
    )
    parser.add_argument(
        "--keep-only-tasks",
        nargs="*",
        choices=["jungle", "support", "team"],
        default=None,
        help="Opcional. Limita los filtros/chequeos a un subconjunto de tareas.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Fracción (ej 0.1) o lee TFG_SAMPLE_FRAC para ajustar rutas por defecto.",
    )
    parser.add_argument(
        "--label-max-minute",
        type=float,
        default=None,
        help="Fase 1: carga labels generadas para una ventana concreta (ej. 8 -> jungle_labels_m08.parquet).",
    )
    parser.add_argument(
        "--label-schema",
        choices=["ternary", "binary_clean", "binary_full"],
        default="ternary",
        help="Fase 2: etiqueta el model_input según el esquema de discretización usado al construir labels.",
    )
    parser.add_argument("--lower-quantile", type=float, default=0.20,
                        help="Solo para resolver sufijos de experimento y trazabilidad.")
    parser.add_argument("--upper-quantile", type=float, default=0.80,
                        help="Solo para resolver sufijos de experimento y trazabilidad.")
    parser.add_argument("--lower-threshold", type=float, default=None,
                        help="Solo para resolver sufijos de experimento y trazabilidad.")
    parser.add_argument("--upper-threshold", type=float, default=None,
                        help="Solo para resolver sufijos de experimento y trazabilidad.")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def apply_sample_suffix(path: Optional[str], frac: Optional[float]) -> Optional[str]:
    if path is None or frac is None or frac >= 1.0 or frac <= 0.0:
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


def select_label_columns(label_df: pd.DataFrame, draft_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in label_df.columns if c not in draft_df.columns or c in JOIN_KEYS]
    return label_df[cols].copy()


def available_tasks_from_paths(args: argparse.Namespace) -> List[str]:
    tasks = []
    if args.jungle_labels_path:
        tasks.append("jungle")
    if args.support_labels_path:
        tasks.append("support")
    if args.team_labels_path:
        tasks.append("team")
    return tasks


def effective_tasks(args: argparse.Namespace) -> List[str]:
    available = available_tasks_from_paths(args)
    if args.keep_only_tasks:
        wanted = [t for t in args.keep_only_tasks if t in available]
        return wanted
    return available


def add_coverage_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["has_jungle_score"] = out["jungle_presence_score"].notna() if "jungle_presence_score" in out.columns else False
    out["has_support_score"] = out["support_roam_score"].notna() if "support_roam_score" in out.columns else False
    out["has_team_score"] = out["team_side_focus_score"].notna() if "team_side_focus_score" in out.columns else False

    out["has_jungle_label"] = out[JUNGLE_LABEL_COL].notna() if JUNGLE_LABEL_COL in out.columns else False
    out["has_support_label"] = out[SUPPORT_LABEL_COL].notna() if SUPPORT_LABEL_COL in out.columns else False
    out["has_team_label"] = out[TEAM_LABEL_COL].notna() if TEAM_LABEL_COL in out.columns else False

    out["has_all_three_scores"] = (
        out["has_jungle_score"] & out["has_support_score"] & out["has_team_score"]
    )
    out["has_all_three_labels"] = (
        out["has_jungle_label"] & out["has_support_label"] & out["has_team_label"]
    )
    return out


def filter_requirements(df: pd.DataFrame, tasks: List[str], require_all_three_scores: bool, require_all_three_labels: bool) -> pd.DataFrame:
    out = df.copy()
    if require_all_three_scores:
        if set(tasks) != {"jungle", "support", "team"}:
            raise SystemExit("--require-all-three-scores requiere haber cargado jungle, support y team labels.")
        out = out[out["has_all_three_scores"]].copy()
    if require_all_three_labels:
        if set(tasks) != {"jungle", "support", "team"}:
            raise SystemExit("--require-all-three-labels requiere haber cargado jungle, support y team labels.")
        out = out[out["has_all_three_labels"]].copy()
    return out


def drop_ambiguous_rows(df: pd.DataFrame, tasks: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = df.copy()
    removed: Dict[str, int] = {}
    for task in tasks:
        label_col, ambiguous_value = AMBIGUOUS_BY_TASK[task]
        if label_col not in out.columns:
            continue
        before = len(out)
        out = out[out[label_col].isna() | (out[label_col] != ambiguous_value)].copy()
        removed[task] = before - len(out)
    return out, removed


def build_task_coverage_summary(model_df: pd.DataFrame) -> pd.DataFrame:
    rows = [{
        "rows": len(model_df),
        "unique_match_team_keys": int(model_df[JOIN_KEYS].drop_duplicates().shape[0]),
        "has_jungle_score": int(model_df["has_jungle_score"].sum()) if "has_jungle_score" in model_df.columns else 0,
        "has_support_score": int(model_df["has_support_score"].sum()) if "has_support_score" in model_df.columns else 0,
        "has_team_score": int(model_df["has_team_score"].sum()) if "has_team_score" in model_df.columns else 0,
        "has_jungle_label": int(model_df["has_jungle_label"].sum()) if "has_jungle_label" in model_df.columns else 0,
        "has_support_label": int(model_df["has_support_label"].sum()) if "has_support_label" in model_df.columns else 0,
        "has_team_label": int(model_df["has_team_label"].sum()) if "has_team_label" in model_df.columns else 0,
        "has_all_three_scores": int(model_df["has_all_three_scores"].sum()) if "has_all_three_scores" in model_df.columns else 0,
        "has_all_three_labels": int(model_df["has_all_three_labels"].sum()) if "has_all_three_labels" in model_df.columns else 0,
    }]
    return pd.DataFrame(rows)


def build_label_counts(model_df: pd.DataFrame, label_col: str, task: str) -> pd.DataFrame:
    if label_col not in model_df.columns:
        return pd.DataFrame()
    counts = (
        model_df.groupby(label_col, dropna=False)
        .size()
        .reset_index(name="n")
        .rename(columns={label_col: "label_value"})
    )
    counts.insert(0, "task", task)
    return counts.sort_values("n", ascending=False).reset_index(drop=True)


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
) -> None:
    ensure_dir(summary_dir)

    overlap_parts: List[pd.DataFrame] = []
    for task_name, task_df in task_tables.items():
        overlap_parts.append(build_overlap_summary(draft_df, task_df, task_name))
    overlap_summary = pd.concat(overlap_parts, ignore_index=True) if overlap_parts else pd.DataFrame()
    if not overlap_summary.empty:
        save_df(overlap_summary, str(Path(summary_dir) / "join_overlap_summary"))

    overall_summary = pd.DataFrame([
        {
            "model_rows": len(model_df),
            "model_unique_matches": int(model_df["match_id"].nunique()) if "match_id" in model_df.columns else None,
            "model_unique_teams": int(model_df["team_id"].nunique()) if "team_id" in model_df.columns else None,
            "model_unique_match_team_keys": int(model_df[JOIN_KEYS].drop_duplicates().shape[0]),
            "label_window_tag": model_df["window_tag"].dropna().iloc[0] if "window_tag" in model_df.columns and model_df["window_tag"].notna().any() else None,
            "label_max_minute": float(model_df["max_minute"].dropna().iloc[0]) if "max_minute" in model_df.columns and model_df["max_minute"].notna().any() else None,
            "label_schema": model_df["label_schema"].dropna().iloc[0] if "label_schema" in model_df.columns and model_df["label_schema"].notna().any() else None,
            "quantile_or_threshold_tag": model_df["quantile_or_threshold_tag"].dropna().iloc[0] if "quantile_or_threshold_tag" in model_df.columns and model_df["quantile_or_threshold_tag"].notna().any() else None,
        }
    ])
    save_df(overall_summary, str(Path(summary_dir) / "overall_model_input_summary"))

    coverage_summary = build_task_coverage_summary(model_df)
    save_df(coverage_summary, str(Path(summary_dir) / "task_coverage_summary"))

    label_counts_parts = []
    label_counts_parts.append(build_label_counts(model_df, JUNGLE_LABEL_COL, "jungle"))
    label_counts_parts.append(build_label_counts(model_df, SUPPORT_LABEL_COL, "support"))
    label_counts_parts.append(build_label_counts(model_df, TEAM_LABEL_COL, "team"))
    label_counts = pd.concat([df for df in label_counts_parts if not df.empty], ignore_index=True) if any(not df.empty for df in label_counts_parts) else pd.DataFrame()
    if not label_counts.empty:
        save_df(label_counts, str(Path(summary_dir) / "label_counts_by_task"))

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


def main() -> None:
    args = parse_args()

    target_frac = get_target_frac(args.sample_frac)
    if target_frac is not None and 0.0 < target_frac < 1.0:
        draft_path = apply_sample_suffix(DEFAULT_DRAFT_PATH, target_frac)
        jungle_labels_path = apply_sample_suffix(DEFAULT_JUNGLE_LABELS_PATH, target_frac)
        support_labels_path = apply_sample_suffix(DEFAULT_SUPPORT_LABELS_PATH, target_frac)
        team_labels_path = apply_sample_suffix(DEFAULT_TEAM_LABELS_PATH, target_frac)
        out_path_str = apply_sample_suffix(DEFAULT_OUT_PATH, target_frac)
        print(f"Muestreo detectado ({target_frac}). Rutas ajustadas automáticamente a sufijos _sample.")
    else:
        draft_path = DEFAULT_DRAFT_PATH
        jungle_labels_path = DEFAULT_JUNGLE_LABELS_PATH
        support_labels_path = DEFAULT_SUPPORT_LABELS_PATH
        team_labels_path = DEFAULT_TEAM_LABELS_PATH
        out_path_str = DEFAULT_OUT_PATH

    if args.label_max_minute is not None:
        jungle_labels_path = apply_window_suffix(jungle_labels_path, args.label_max_minute)
        support_labels_path = apply_window_suffix(support_labels_path, args.label_max_minute)
        team_labels_path = apply_window_suffix(team_labels_path, args.label_max_minute)
        out_path_str = apply_window_suffix(out_path_str, args.label_max_minute)
        print(f"Fase 1: cargando labels para ventana 0-{args.label_max_minute:g} min ({format_window_tag(args.label_max_minute)}).")

    quantile_tag = format_quantile_or_threshold_tag(
        args.lower_quantile, args.upper_quantile, args.lower_threshold, args.upper_threshold
    )
    jungle_labels_path = apply_quantile_suffix(
        jungle_labels_path, args.lower_quantile, args.upper_quantile, args.lower_threshold, args.upper_threshold
    )
    support_labels_path = apply_quantile_suffix(
        support_labels_path, args.lower_quantile, args.upper_quantile, args.lower_threshold, args.upper_threshold
    )
    team_labels_path = apply_quantile_suffix(
        team_labels_path, args.lower_quantile, args.upper_quantile, args.lower_threshold, args.upper_threshold
    )
    out_path_str = apply_quantile_suffix(
        out_path_str, args.lower_quantile, args.upper_quantile, args.lower_threshold, args.upper_threshold
    )
    out_path_str = apply_schema_suffix(out_path_str, args.label_schema)
    print(f"Fase 2: esquema de labels = {args.label_schema}.")
    print(f"Fase 2: tag de quantiles/thresholds = {quantile_tag}.")

    task_names = [t for t, p in [("jungle", jungle_labels_path), ("support", support_labels_path), ("team", team_labels_path)] if p]
    if not task_names:
        raise SystemExit("Debes proporcionar al menos una tabla de labels. Edita DEFAULT_JUNGLE/SUPPORT/TEAM_LABELS_PATH en el script.")

    selected_tasks = [t for t in (args.keep_only_tasks or task_names) if t in task_names]

    out_path = Path(out_path_str)
    ensure_dir(str(out_path.parent))

    summary_dir = DEFAULT_SUMMARY_DIR or str(out_path.with_suffix("")) + "_analysis"

    print("\n[Rutas] Cargando draft features (Entrada):", os.path.abspath(draft_path))
    draft_df = load_parquet(draft_path)
    validate_required_columns(draft_df, "draft_features", JOIN_KEYS)
    validate_no_duplicate_keys(draft_df, "draft_features")

    model_df = draft_df.copy()
    loaded_task_tables: Dict[str, pd.DataFrame] = {}

    task_specs = [
        ("jungle", jungle_labels_path, JUNGLE_REQUIRED_ANY + [JUNGLE_LABEL_COL]),
        ("support", support_labels_path, SUPPORT_REQUIRED_ANY + [SUPPORT_LABEL_COL]),
        ("team", team_labels_path, TEAM_REQUIRED_ANY + [TEAM_LABEL_COL]),
    ]

    for task_name, path, required_cols in task_specs:
        if not path:
            continue
        if not Path(path).exists():
            print(f"[WARN] {task_name} labels no encontrado: {os.path.abspath(path)} — se omite esta tarea.")
            continue
        print(f"[Rutas] Cargando {task_name} labels (Entrada): {os.path.abspath(path)}")
        task_df = load_parquet(path)
        validate_required_columns(task_df, f"{task_name}_labels", JOIN_KEYS)
        validate_required_columns(task_df, f"{task_name}_labels", required_cols)
        validate_no_duplicate_keys(task_df, f"{task_name}_labels")
        task_df = select_label_columns(task_df, draft_df)
        task_df["label_schema"] = args.label_schema
        task_df["quantile_or_threshold_tag"] = quantile_tag
        loaded_task_tables[task_name] = task_df.copy()
        model_df = model_df.merge(task_df, on=JOIN_KEYS, how=args.join_how, validate="one_to_one")
        print(f"  - Filas tras unir {task_name}: {len(model_df)}")

    if not loaded_task_tables:
        raise SystemExit(
            "No se ha cargado ninguna tabla de labels. El model_input quedaria sin targets. Revisa las rutas de labels, la ventana y el esquema."
        )
    missing_selected = [t for t in selected_tasks if t not in loaded_task_tables]
    if missing_selected:
        raise SystemExit(
            f"Faltan tablas de labels para las tareas seleccionadas: {missing_selected}. No se genera model_input incompleto."
        )

    model_df = add_coverage_columns(model_df)
    model_df = filter_requirements(
        model_df,
        tasks=selected_tasks,
        require_all_three_scores=args.require_all_three_scores,
        require_all_three_labels=args.require_all_three_labels,
    )

    removed_ambiguous: Dict[str, int] = {}
    if args.drop_ambiguous_labels:
        before = len(model_df)
        model_df, removed_ambiguous = drop_ambiguous_rows(model_df, selected_tasks)
        print(f"Filas tras drop ambiguous: {len(model_df)} (eliminadas {before - len(model_df)})")
        if removed_ambiguous:
            print("Eliminadas por tarea:", removed_ambiguous)

    validate_no_duplicate_keys(model_df, "model_input_multioutput")

    print(f"\n[Rutas] model_input parquet se guardará en (Salida): {os.path.abspath(out_path)}")
    print(f"[Rutas] Reportes de unión y análisis se guardarán en (Salida): {os.path.abspath(summary_dir)}\n")

    print(f"Draft rows: {len(draft_df)}")
    for task_name, task_df in loaded_task_tables.items():
        print(f"{task_name.title()} rows: {len(task_df)}")
    print(f"Filas finales model_input: {len(model_df)}")

    print("\n[DEBUG] Columnas finales del model_input:")
    print(sorted(model_df.columns.tolist()))
    expected_targets = [JUNGLE_LABEL_COL, SUPPORT_LABEL_COL, TEAM_LABEL_COL]
    for col in expected_targets:
        print(f"[DEBUG] {col}: {'OK' if col in model_df.columns else 'MISSING'}")

    model_df.to_parquet(out_path, index=False)
    write_summary_tables(draft_df, loaded_task_tables, model_df, summary_dir)

    print("\nHecho.")
    print(f"- model input parquet: {out_path}")
    print(f"- analysis dir: {summary_dir}")


if __name__ == "__main__":
    main()
