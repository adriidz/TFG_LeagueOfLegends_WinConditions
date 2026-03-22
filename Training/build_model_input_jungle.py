
#!/usr/bin/env python3
"""
build_model_input_jungle.py

Une:
- draft_features.parquet
- jungle_labels.parquet

por:
- match_id
- team_id

y genera un parquet final listo para entrenamiento del modelo de jungla.
"""

import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd


JOIN_KEYS = ["match_id", "team_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye model_input_jungle.parquet uniendo draft features y labels."
    )
    parser.add_argument("--draft-path", default=os.path.join("Data_clean", "features", "draft_features.parquet"), help="Ruta al parquet de draft features.")
    parser.add_argument("--labels-path", default=os.path.join("Data_clean", "labels", "jungle_labels.parquet"), help="Ruta al parquet de labels de jungla.")
    parser.add_argument("--out-path", default=os.path.join("Data_training", "model_input_jungle.parquet"), help="Ruta de salida del parquet unido.")
    parser.add_argument(
        "--summary-dir",
        default=None,
        help="Directorio opcional para guardar resúmenes de validación.",
    )
    parser.add_argument(
        "--drop-ambiguous",
        action="store_true",
        help="Si se activa, elimina filas con jungle_presence_label='ambiguous'.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_parquet(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    if p.suffix.lower() != ".parquet":
        raise ValueError(f"Se esperaba un .parquet: {path}")
    return pd.read_parquet(p)


def validate_required_columns(df: pd.DataFrame, name: str, required_cols: List[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en {name}: {missing}")


def validate_no_duplicate_keys(df: pd.DataFrame, name: str) -> None:
    dup_mask = df.duplicated(subset=JOIN_KEYS, keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows > 0:
        dup_preview = df.loc[dup_mask, JOIN_KEYS].head(10)
        raise ValueError(
            f"Se han encontrado {n_dup_rows} filas duplicadas por {JOIN_KEYS} en {name}.\n"
            f"Primeros duplicados:\n{dup_preview.to_string(index=False)}"
        )


def build_overlap_summary(draft_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    draft_keys = draft_df[JOIN_KEYS].drop_duplicates().copy()
    labels_keys = labels_df[JOIN_KEYS].drop_duplicates().copy()

    draft_key_set = set(map(tuple, draft_keys.to_numpy()))
    labels_key_set = set(map(tuple, labels_keys.to_numpy()))

    overlap = draft_key_set & labels_key_set
    draft_only = draft_key_set - labels_key_set
    labels_only = labels_key_set - draft_key_set

    return pd.DataFrame(
        [
            {
                "draft_rows": len(draft_df),
                "labels_rows": len(labels_df),
                "draft_unique_keys": len(draft_key_set),
                "labels_unique_keys": len(labels_key_set),
                "overlap_keys": len(overlap),
                "draft_only_keys": len(draft_only),
                "labels_only_keys": len(labels_only),
            }
        ]
    )


def write_summary_tables(
    draft_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    model_df: pd.DataFrame,
    summary_dir: str,
) -> None:
    ensure_dir(summary_dir)

    overlap_summary = build_overlap_summary(draft_df, labels_df)
    overlap_summary.to_parquet(Path(summary_dir) / "join_overlap_summary.parquet", index=False)
    overlap_summary.to_csv(Path(summary_dir) / "join_overlap_summary.csv", index=False)

    overall_summary = pd.DataFrame(
        [
            {
                "model_rows": len(model_df),
                "model_unique_matches": int(model_df["match_id"].nunique()) if "match_id" in model_df.columns else None,
                "model_unique_teams": int(model_df["team_id"].nunique()) if "team_id" in model_df.columns else None,
                "model_unique_match_team_keys": int(model_df[JOIN_KEYS].drop_duplicates().shape[0]),
                "ambiguous_rows": int((model_df["jungle_presence_label"] == "ambiguous").sum())
                if "jungle_presence_label" in model_df.columns else None,
            }
        ]
    )
    overall_summary.to_parquet(Path(summary_dir) / "overall_model_input_summary.parquet", index=False)
    overall_summary.to_csv(Path(summary_dir) / "overall_model_input_summary.csv", index=False)

    if "jungle_presence_label" in model_df.columns:
        label_counts = (
            model_df.groupby("jungle_presence_label", dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
        )
        label_counts.to_parquet(Path(summary_dir) / "label_counts.parquet", index=False)
        label_counts.to_csv(Path(summary_dir) / "label_counts.csv", index=False)

    if "patch" in model_df.columns:
        patch_counts = (
            model_df.groupby("patch", dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
        )
        patch_counts.to_parquet(Path(summary_dir) / "patch_counts.parquet", index=False)
        patch_counts.to_csv(Path(summary_dir) / "patch_counts.csv", index=False)

    if "side" in model_df.columns:
        side_counts = (
            model_df.groupby("side", dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
        )
        side_counts.to_parquet(Path(summary_dir) / "side_counts.parquet", index=False)
        side_counts.to_csv(Path(summary_dir) / "side_counts.csv", index=False)


def main() -> None:
    args = parse_args()

    out_path = Path(args.out_path)
    ensure_dir(str(out_path.parent))

    summary_dir = args.summary_dir
    if summary_dir is None:
        summary_dir = str(out_path.with_suffix("")) + "_analysis"

    print("Cargando draft features...")
    draft_df = load_parquet(args.draft_path)

    print("Cargando jungle labels...")
    labels_df = load_parquet(args.labels_path)

    validate_required_columns(draft_df, "draft_features", JOIN_KEYS)
    validate_required_columns(labels_df, "jungle_labels", JOIN_KEYS)
    validate_required_columns(labels_df, "jungle_labels", ["jungle_presence_label"])

    validate_no_duplicate_keys(draft_df, "draft_features")
    validate_no_duplicate_keys(labels_df, "jungle_labels")

    print(f"Draft rows: {len(draft_df)}")
    print(f"Labels rows: {len(labels_df)}")

    label_cols_to_use = [
        c for c in labels_df.columns
        if c not in draft_df.columns or c in JOIN_KEYS
    ]
    labels_df = labels_df[label_cols_to_use].copy()

    print("Haciendo inner join por (match_id, team_id)...")
    model_df = draft_df.merge(
        labels_df,
        on=JOIN_KEYS,
        how="inner",
        validate="one_to_one",
    )

    if args.drop_ambiguous:
        before = len(model_df)
        model_df = model_df[model_df["jungle_presence_label"] != "ambiguous"].copy()
        after = len(model_df)
        print(f"Filas tras drop ambiguous: {after} (eliminadas {before - after})")

    validate_no_duplicate_keys(model_df, "model_input_jungle")

    print(f"Filas finales model_input: {len(model_df)}")
    print(f"Guardando parquet en: {out_path}")
    model_df.to_parquet(out_path, index=False)

    print(f"Guardando resúmenes en: {summary_dir}")
    write_summary_tables(draft_df, labels_df, model_df, summary_dir)

    print("\nHecho.")
    print(f"- model input parquet: {out_path}")
    print(f"- analysis dir: {summary_dir}")


if __name__ == "__main__":
    main()
