#!/usr/bin/env python3
"""
04_analyze_label_stability.py

Analiza la estabilidad de labels entre varias ventanas temporales.

Entrada esperada:
- data/clean/labels/jungle_labels[_sampleXX]_m06.parquet
- data/clean/labels/support_labels[_sampleXX]_m06.parquet
- data/clean/labels/team_tendency_labels[_sampleXX]_m06.parquet
- ... y lo mismo para el resto de ventanas

Salida:
- Un directorio con tablas CSV/Parquet por tarea y resúmenes globales.

Métricas principales:
- agreement exacto entre ventanas consecutivas
- agreement exacto contra una ventana de referencia
- agreement ignorando ambiguous
- flips entre extremos ignorando ambiguous
- número de cambios por fila a lo largo de la secuencia de ventanas
- matrices de transición entre ventanas consecutivas
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

JOIN_KEYS = ["match_id", "team_id"]

TASK_SPECS = {
    "jungle": {
        "prefix": "jungle_labels",
        "label_col": "jungle_presence_label",
        "extremes": ["farm_oriented", "map_presence"],
    },
    "support": {
        "prefix": "support_labels",
        "label_col": "support_roam_label",
        "extremes": ["lane_anchored", "roamer"],
    },
    "team": {
        "prefix": "team_tendency_labels",
        "label_col": "team_tendency_label",
        "extremes": ["botside_oriented", "topside_oriented"],
    },
}

DEFAULT_LABELS_DIR = os.path.join("data", "clean", "labels")
DEFAULT_OUT_BASE = os.path.join("data", "clean", "label_stability_analysis")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analiza estabilidad de labels entre ventanas temporales.")
    p.add_argument("--windows", nargs="+", type=int, required=True,
                   help="Lista de ventanas en minutos. Ej: --windows 6 8 10 12 14")
    p.add_argument("--reference-window", type=int, default=None,
                   help="Ventana de referencia para agreement. Default: última de --windows")
    p.add_argument("--sample-frac", type=float, default=None,
                   help="Fracción de sample para resolver sufijos _sampleXX si aplica.")
    p.add_argument("--labels-dir", default=DEFAULT_LABELS_DIR)
    p.add_argument("--outdir", default=None,
                   help="Directorio de salida. Si no se pasa, se deriva automáticamente.")
    return p.parse_args()


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, path_no_ext: str) -> None:
    ensure_dir(str(Path(path_no_ext).parent))
    df.to_csv(path_no_ext + ".csv", index=False)
    try:
        df.to_parquet(path_no_ext + ".parquet", index=False)
    except Exception:
        pass


def sample_suffix(sample_frac: Optional[float]) -> str:
    if sample_frac is None or sample_frac <= 0.0 or sample_frac >= 1.0:
        return ""
    return f"_sample{int(sample_frac * 100)}"


def window_tag(window: int) -> str:
    return f"m{int(window):02d}"


def build_input_path(labels_dir: str, prefix: str, window: int, sample_frac: Optional[float]) -> str:
    sfx = sample_suffix(sample_frac)
    tag = window_tag(window)
    return os.path.join(labels_dir, f"{prefix}{sfx}_{tag}.parquet")


def build_output_dir(base_out: str, windows: Sequence[int], sample_frac: Optional[float]) -> str:
    sfx = sample_suffix(sample_frac)
    joined = "-".join(window_tag(w) for w in windows)
    return f"{base_out}{sfx}_{joined}"


def load_task_window_df(task_name: str, window: int, labels_dir: str, sample_frac: Optional[float]) -> pd.DataFrame:
    spec = TASK_SPECS[task_name]
    path = build_input_path(labels_dir, spec["prefix"], window, sample_frac)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el parquet para {task_name} ventana {window}: {path}")
    df = pd.read_parquet(path)
    required = JOIN_KEYS + [spec["label_col"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Faltan columnas en {path}: {missing}")
    dup_mask = df.duplicated(subset=JOIN_KEYS, keep=False)
    if dup_mask.any():
        preview = df.loc[dup_mask, JOIN_KEYS].head(10)
        raise SystemExit(
            f"Duplicados por {JOIN_KEYS} en {path}. Primeros:\n{preview.to_string(index=False)}"
        )
    keep = JOIN_KEYS + [spec["label_col"]]
    out = df[keep].copy()
    out = out.rename(columns={spec["label_col"]: f"label_{window_tag(window)}"})
    return out


def build_aligned_table(task_name: str, windows: Sequence[int], labels_dir: str, sample_frac: Optional[float]) -> pd.DataFrame:
    merged: Optional[pd.DataFrame] = None
    for w in windows:
        part = load_task_window_df(task_name, w, labels_dir, sample_frac)
        if merged is None:
            merged = part
        else:
            merged = merged.merge(part, on=JOIN_KEYS, how="outer", validate="one_to_one")
    assert merged is not None
    return merged.sort_values(JOIN_KEYS).reset_index(drop=True)


def exact_agreement(a: pd.Series, b: pd.Series) -> Tuple[int, int, float]:
    valid = a.notna() & b.notna()
    n = int(valid.sum())
    if n == 0:
        return 0, 0, float("nan")
    same = int((a[valid] == b[valid]).sum())
    return n, same, same / n


def non_ambiguous_subset(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.notna() & b.notna() & (a != "ambiguous") & (b != "ambiguous")


def non_ambiguous_agreement(a: pd.Series, b: pd.Series) -> Tuple[int, int, float]:
    valid = non_ambiguous_subset(a, b)
    n = int(valid.sum())
    if n == 0:
        return 0, 0, float("nan")
    same = int((a[valid] == b[valid]).sum())
    return n, same, same / n


def extreme_flip_rate(a: pd.Series, b: pd.Series, extremes: Sequence[str]) -> Tuple[int, int, float]:
    valid = non_ambiguous_subset(a, b) & a.isin(extremes) & b.isin(extremes)
    n = int(valid.sum())
    if n == 0:
        return 0, 0, float("nan")
    flips = int((a[valid] != b[valid]).sum())
    return n, flips, flips / n


def build_pairwise_agreement(task_name: str, aligned: pd.DataFrame, windows: Sequence[int], reference_window: int) -> pd.DataFrame:
    rows: List[dict] = []
    spec = TASK_SPECS[task_name]
    extremes = spec["extremes"]

    pairs: List[Tuple[int, int, str]] = []
    for i in range(len(windows) - 1):
        pairs.append((windows[i], windows[i + 1], "consecutive"))
    for w in windows:
        if w != reference_window:
            pairs.append((w, reference_window, "vs_reference"))

    for w1, w2, relation in pairs:
        c1 = f"label_{window_tag(w1)}"
        c2 = f"label_{window_tag(w2)}"
        n_exact, same_exact, rate_exact = exact_agreement(aligned[c1], aligned[c2])
        n_nonamb, same_nonamb, rate_nonamb = non_ambiguous_agreement(aligned[c1], aligned[c2])
        n_flip, flips, flip_rate = extreme_flip_rate(aligned[c1], aligned[c2], extremes)
        rows.append({
            "task": task_name,
            "window_a": w1,
            "window_b": w2,
            "relation": relation,
            "n_exact": n_exact,
            "same_exact": same_exact,
            "agreement_exact": rate_exact,
            "n_non_ambiguous": n_nonamb,
            "same_non_ambiguous": same_nonamb,
            "agreement_non_ambiguous": rate_nonamb,
            "n_extreme_pairs": n_flip,
            "extreme_flips": flips,
            "extreme_flip_rate": flip_rate,
        })
    return pd.DataFrame(rows)


def count_changes_across_windows(values: List[object]) -> int:
    changes = 0
    for a, b in zip(values[:-1], values[1:]):
        if pd.isna(a) or pd.isna(b):
            continue
        if a != b:
            changes += 1
    return changes


def count_extreme_flips_across_windows(values: List[object], extremes: Sequence[str]) -> int:
    flips = 0
    for a, b in zip(values[:-1], values[1:]):
        if pd.isna(a) or pd.isna(b):
            continue
        if a == "ambiguous" or b == "ambiguous":
            continue
        if a in extremes and b in extremes and a != b:
            flips += 1
    return flips


def build_change_tables(task_name: str, aligned: pd.DataFrame, windows: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    spec = TASK_SPECS[task_name]
    extremes = spec["extremes"]
    label_cols = [f"label_{window_tag(w)}" for w in windows]

    detail = aligned.copy()
    detail["label_path"] = detail[label_cols].astype(str).agg(" | ".join, axis=1)
    detail["n_observed_windows"] = detail[label_cols].notna().sum(axis=1)
    detail["n_label_changes"] = detail[label_cols].apply(lambda r: count_changes_across_windows(r.tolist()), axis=1)
    detail["n_extreme_flips_ignore_ambiguous"] = detail[label_cols].apply(
        lambda r: count_extreme_flips_across_windows(r.tolist(), extremes), axis=1
    )
    detail["is_fully_stable"] = detail["n_label_changes"] == 0
    detail["has_any_change"] = detail["n_label_changes"] > 0

    summary = pd.DataFrame([{
        "task": task_name,
        "rows": len(detail),
        "fully_observed_rows": int((detail["n_observed_windows"] == len(windows)).sum()),
        "fully_stable_rows": int(detail["is_fully_stable"].sum()),
        "rows_with_any_change": int(detail["has_any_change"].sum()),
        "pct_fully_stable": float(detail["is_fully_stable"].mean()) if len(detail) else float("nan"),
        "mean_n_label_changes": float(detail["n_label_changes"].mean()) if len(detail) else float("nan"),
        "median_n_label_changes": float(detail["n_label_changes"].median()) if len(detail) else float("nan"),
        "rows_with_extreme_flip_ignore_ambiguous": int((detail["n_extreme_flips_ignore_ambiguous"] > 0).sum()),
        "pct_rows_with_extreme_flip_ignore_ambiguous": (
            float((detail["n_extreme_flips_ignore_ambiguous"] > 0).mean()) if len(detail) else float("nan")
        ),
    }])
    return detail, summary


def build_transition_tables(task_name: str, aligned: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for w1, w2 in zip(windows[:-1], windows[1:]):
        c1 = f"label_{window_tag(w1)}"
        c2 = f"label_{window_tag(w2)}"
        work = aligned[[c1, c2]].copy()
        work = work.rename(columns={c1: "from_label", c2: "to_label"})
        work = work[work["from_label"].notna() & work["to_label"].notna()].copy()
        if work.empty:
            continue
        trans = work.groupby(["from_label", "to_label"]).size().reset_index(name="n")
        trans["task"] = task_name
        trans["window_a"] = w1
        trans["window_b"] = w2
        total = trans["n"].sum()
        trans["pct_of_pair"] = trans["n"] / total if total > 0 else float("nan")
        rows.append(trans[["task", "window_a", "window_b", "from_label", "to_label", "n", "pct_of_pair"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["task", "window_a", "window_b", "from_label", "to_label", "n", "pct_of_pair"]
    )


def main() -> None:
    args = parse_args()
    windows = sorted(dict.fromkeys(args.windows))
    if len(windows) < 2:
        raise SystemExit("Debes pasar al menos dos ventanas en --windows.")

    reference_window = args.reference_window if args.reference_window is not None else windows[-1]
    if reference_window not in windows:
        raise SystemExit("--reference-window debe estar incluido en --windows.")

    outdir = args.outdir or build_output_dir(DEFAULT_OUT_BASE, windows, args.sample_frac)
    ensure_dir(outdir)

    task_pairwise_parts: List[pd.DataFrame] = []
    task_change_parts: List[pd.DataFrame] = []
    task_transition_parts: List[pd.DataFrame] = []

    for task_name in TASK_SPECS:
        print(f"[Task] Analizando {task_name}...")
        task_dir = os.path.join(outdir, task_name)
        ensure_dir(task_dir)

        aligned = build_aligned_table(task_name, windows, args.labels_dir, args.sample_frac)
        pairwise = build_pairwise_agreement(task_name, aligned, windows, reference_window)
        change_detail, change_summary = build_change_tables(task_name, aligned, windows)
        transitions = build_transition_tables(task_name, aligned, windows)

        save_df(aligned, os.path.join(task_dir, f"{task_name}_labels_aligned"))
        save_df(pairwise, os.path.join(task_dir, f"{task_name}_pairwise_agreement"))
        save_df(change_detail, os.path.join(task_dir, f"{task_name}_change_detail"))
        save_df(change_summary, os.path.join(task_dir, f"{task_name}_change_summary"))
        save_df(transitions, os.path.join(task_dir, f"{task_name}_transitions"))

        task_pairwise_parts.append(pairwise)
        task_change_parts.append(change_summary)
        task_transition_parts.append(transitions)

    all_pairwise = pd.concat(task_pairwise_parts, ignore_index=True) if task_pairwise_parts else pd.DataFrame()
    all_changes = pd.concat(task_change_parts, ignore_index=True) if task_change_parts else pd.DataFrame()
    all_transitions = pd.concat(task_transition_parts, ignore_index=True) if task_transition_parts else pd.DataFrame()

    save_df(all_pairwise, os.path.join(outdir, "all_tasks_pairwise_agreement"))
    save_df(all_changes, os.path.join(outdir, "all_tasks_change_summary"))
    save_df(all_transitions, os.path.join(outdir, "all_tasks_transitions"))

    run_summary = pd.DataFrame([{
        "windows": ",".join(str(w) for w in windows),
        "reference_window": reference_window,
        "sample_frac": args.sample_frac,
        "labels_dir": os.path.abspath(args.labels_dir),
        "outdir": os.path.abspath(outdir),
    }])
    save_df(run_summary, os.path.join(outdir, "run_summary"))

    print("\nHecho.")
    print(f"- Output dir: {outdir}")


if __name__ == "__main__":
    main()
