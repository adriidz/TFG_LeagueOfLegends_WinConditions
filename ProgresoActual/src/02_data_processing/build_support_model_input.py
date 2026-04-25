#!/usr/bin/env python3
"""
Build the support-only model input for continuous regression.

This is the clean ProgresoActual replacement for the old multi-output builder.
It joins:

    draft_features + support_scores -> model_input_support_regression

No jungle/team targets are required.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

JOIN_KEYS = ["match_id", "team_id"]
SUPPORT_SCORE_COL = "support_roam_score"

DEFAULT_DRAFT_PATH = os.path.join("ProgresoActual", "data", "clean", "features", "draft_features.parquet")
DEFAULT_SUPPORT_SCORES_PATH = os.path.join("ProgresoActual", "data", "clean", "scores", "support_scores.parquet")
DEFAULT_OUT_PATH = os.path.join("ProgresoActual", "data", "training", "model_input_support_regression.parquet")
DEFAULT_SUMMARY_DIR = None


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def format_window_tag(max_minute: float) -> str:
    return f"m{int(round(float(max_minute))):02d}"


def apply_window_suffix(path: Optional[str], max_minute: Optional[float]) -> Optional[str]:
    if path is None or max_minute is None:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_{format_window_tag(max_minute)}{ext}"


def apply_sample_suffix(path: Optional[str], frac: Optional[float]) -> Optional[str]:
    if path is None or frac is None or frac <= 0.0 or frac >= 1.0:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_sample{int(round(frac * 100))}{ext}"


def append_suffix_to_dir(path: Optional[str], suffix: Optional[str]) -> Optional[str]:
    if path is None or not suffix:
        return path
    return f"{path}_{suffix}"


def get_target_frac(args_frac: Optional[float]) -> Optional[float]:
    if args_frac is not None:
        return args_frac
    env_frac = os.getenv("TFG_SAMPLE_FRAC")
    if env_frac:
        try:
            return float(env_frac)
        except ValueError:
            return None
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build support-only regression model input.")
    p.add_argument("--draft-path", default=None)
    p.add_argument("--support-scores-path", default=None)
    p.add_argument("--out-path", default=None)
    p.add_argument("--summary-dir", default=None)
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--score-max-minute", type=float, default=None)
    p.add_argument("--join-how", choices=["left", "inner"], default="left")
    p.add_argument("--allow-missing-support-score", action="store_true")
    p.add_argument(
        "--support-score-source-col",
        default="support_roam_score_v2",
        help="Source score column in support_scores. Renamed to support_roam_score in output.",
    )
    p.add_argument(
        "--support-extra-cols",
        nargs="*",
        default=[
            "support_score_confidence_v2",
            "support_adc_xp_ratio_v2",
            "mean_distance_to_adc_v2",
            "outside_ratio",
            "far_ratio",
            "xp_gap",
            "valid_support_frames_v2",
            "valid_coop_frames_v2",
            "config_id",
            "start_minute",
            "max_minute",
            "far_adc_threshold",
            "w_outside",
            "w_far",
            "w_xp",
            "window_tag",
        ],
    )
    return p.parse_args()


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
    dup = df.duplicated(subset=JOIN_KEYS, keep=False)
    if dup.any():
        preview = df.loc[dup, JOIN_KEYS].head(10)
        raise SystemExit(
            f"Se han encontrado duplicados por {JOIN_KEYS} en {name}.\n"
            f"Primeros duplicados:\n{preview.to_string(index=False)}"
        )


def resolve_paths(args: argparse.Namespace) -> Tuple[str, str, str, str]:
    target_frac = get_target_frac(args.sample_frac)
    draft_path = args.draft_path or DEFAULT_DRAFT_PATH
    support_scores_path = args.support_scores_path or DEFAULT_SUPPORT_SCORES_PATH
    out_path = args.out_path or DEFAULT_OUT_PATH
    summary_dir = args.summary_dir or DEFAULT_SUMMARY_DIR

    if target_frac is not None and 0.0 < target_frac < 1.0:
        draft_path = apply_sample_suffix(draft_path, target_frac)
        support_scores_path = apply_sample_suffix(support_scores_path, target_frac)
        out_path = apply_sample_suffix(out_path, target_frac)

    if args.score_max_minute is not None:
        support_scores_path = apply_window_suffix(support_scores_path, args.score_max_minute)
        out_path = apply_window_suffix(out_path, args.score_max_minute)
        if summary_dir:
            summary_dir = append_suffix_to_dir(summary_dir, format_window_tag(args.score_max_minute))

    if summary_dir is None:
        summary_dir = str(Path(out_path).with_suffix("")) + "_analysis"
    return draft_path, support_scores_path, out_path, summary_dir


def select_support_columns(score_df: pd.DataFrame, score_col: str, extras: List[str]) -> pd.DataFrame:
    keep = JOIN_KEYS + [score_col]
    keep.extend([c for c in extras if c in score_df.columns])
    keep = list(dict.fromkeys([c for c in keep if c in score_df.columns]))
    return score_df[keep].copy()


def save_df(df: pd.DataFrame, path_no_ext: str) -> None:
    ensure_dir(str(Path(path_no_ext).parent))
    df.to_csv(path_no_ext + ".csv", index=False)
    try:
        df.to_parquet(path_no_ext + ".parquet", index=False)
    except Exception:
        pass


def build_overlap_summary(draft_df: pd.DataFrame, support_df: pd.DataFrame) -> pd.DataFrame:
    draft_keys = set(map(tuple, draft_df[JOIN_KEYS].drop_duplicates().to_numpy()))
    support_keys = set(map(tuple, support_df[JOIN_KEYS].drop_duplicates().to_numpy()))
    return pd.DataFrame([{
        "table": "support_scores",
        "draft_rows": int(len(draft_df)),
        "support_rows": int(len(support_df)),
        "draft_unique_keys": int(len(draft_keys)),
        "support_unique_keys": int(len(support_keys)),
        "overlap_keys": int(len(draft_keys & support_keys)),
        "draft_only_keys": int(len(draft_keys - support_keys)),
        "support_only_keys": int(len(support_keys - draft_keys)),
    }])


def write_summary_tables(
    draft_df: pd.DataFrame,
    support_df: pd.DataFrame,
    model_df: pd.DataFrame,
    summary_dir: str,
    support_source_col: str,
) -> None:
    ensure_dir(summary_dir)
    save_df(build_overlap_summary(draft_df, support_df), str(Path(summary_dir) / "join_overlap_summary"))

    overall = pd.DataFrame([{
        "model_rows": int(len(model_df)),
        "model_unique_matches": int(model_df["match_id"].nunique()) if "match_id" in model_df.columns else 0,
        "model_unique_match_team_keys": int(model_df[JOIN_KEYS].drop_duplicates().shape[0]),
        "has_support_score": int(model_df["has_support_score"].sum()) if "has_support_score" in model_df.columns else 0,
        "support_score_source_col": support_source_col,
        "window_tag": model_df["window_tag"].dropna().iloc[0] if "window_tag" in model_df.columns and model_df["window_tag"].notna().any() else None,
        "score_max_minute": float(model_df["max_minute"].dropna().iloc[0]) if "max_minute" in model_df.columns and model_df["max_minute"].notna().any() else None,
    }])
    save_df(overall, str(Path(summary_dir) / "overall_model_input_summary"))

    if SUPPORT_SCORE_COL in model_df.columns:
        valid = model_df[SUPPORT_SCORE_COL].dropna()
        if not valid.empty:
            score_summary = pd.DataFrame([{
                "score_col": SUPPORT_SCORE_COL,
                "n": int(valid.shape[0]),
                "mean": float(valid.mean()),
                "std": float(valid.std()),
                "min": float(valid.min()),
                "q25": float(valid.quantile(0.25)),
                "median": float(valid.quantile(0.50)),
                "q75": float(valid.quantile(0.75)),
                "max": float(valid.max()),
            }])
            save_df(score_summary, str(Path(summary_dir) / "support_score_summary"))

    if "patch" in model_df.columns:
        patch_counts = model_df.groupby("patch", dropna=False).size().reset_index(name="n").sort_values("n", ascending=False)
        save_df(patch_counts, str(Path(summary_dir) / "patch_counts"))
    if "side" in model_df.columns:
        side_counts = model_df.groupby("side", dropna=False).size().reset_index(name="n").sort_values("side")
        save_df(side_counts, str(Path(summary_dir) / "side_counts"))


def main() -> None:
    args = parse_args()
    draft_path, support_scores_path, out_path, summary_dir = resolve_paths(args)

    print(f"[Input] draft_features: {os.path.abspath(draft_path)}")
    draft_df = load_parquet(draft_path)
    validate_required_columns(draft_df, "draft_features", JOIN_KEYS)
    validate_no_duplicate_keys(draft_df, "draft_features")

    print(f"[Input] support_scores: {os.path.abspath(support_scores_path)}")
    support_df = load_parquet(support_scores_path)
    validate_required_columns(support_df, "support_scores", JOIN_KEYS + [args.support_score_source_col])
    validate_no_duplicate_keys(support_df, "support_scores")

    support_df = select_support_columns(support_df, args.support_score_source_col, args.support_extra_cols)
    if args.support_score_source_col != SUPPORT_SCORE_COL:
        support_df = support_df.rename(columns={args.support_score_source_col: SUPPORT_SCORE_COL})

    model_df = draft_df.merge(support_df, on=JOIN_KEYS, how=args.join_how, validate="one_to_one")
    model_df["has_support_score"] = model_df[SUPPORT_SCORE_COL].notna()

    if not args.allow_missing_support_score:
        before = len(model_df)
        model_df = model_df[model_df["has_support_score"]].copy()
        print(f"Filas tras exigir support score: {len(model_df)} (eliminadas {before - len(model_df)})")

    validate_no_duplicate_keys(model_df, "model_input_support_regression")
    out = Path(out_path)
    ensure_dir(str(out.parent))
    ensure_dir(summary_dir)
    model_df = model_df.sort_values(JOIN_KEYS).reset_index(drop=True)
    model_df.to_parquet(out, index=False)
    write_summary_tables(draft_df, support_df, model_df, summary_dir, args.support_score_source_col)

    print(f"\nHecho.")
    print(f"- filas draft: {len(draft_df)}")
    print(f"- filas support_scores: {len(support_df)}")
    print(f"- filas model_input: {len(model_df)}")
    print(f"- parquet: {os.path.abspath(out)}")
    print(f"- analysis: {os.path.abspath(summary_dir)}")


if __name__ == "__main__":
    main()
