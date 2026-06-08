#!/usr/bin/env python3
"""
17_embedding_analysis.py -- Analysis of learned champion embedding space.

Inputs:
  - final/models/mlp_embed/embeddings_raw.json
  - final/models/mlp_per_role/embeddings_raw_ally_utility.json
  - final/data/champion_archetypes.json
  - final/data/champion_classes.json
  - final/data/training/train.parquet (for per-champion mean roam scores)

Outputs:
  - final/analysis/embedding_analysis/embedding_analysis_comparison.csv
  - final/analysis/embedding_analysis/<embedding_label>/nearest_neighbors_all_champions.csv
  - final/analysis/embedding_analysis/<embedding_label>/nearest_neighbors_support_only.csv
  - final/analysis/embedding_analysis/<embedding_label>/embedding_distance_vs_roam.png
  - final/analysis/embedding_analysis/<embedding_label>/embedding_analysis_summary.md

The goal is not to tune the model again, but to inspect whether the learned
champion vectors show measurable neighborhood structure. The report should not
claim semantic structure unless the quantitative checks support it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SHARED_EMBEDDINGS = REPO_ROOT / "final" / "models" / "mlp_embed" / "embeddings_raw.json"
DEFAULT_PER_ROLE_ALLY_UTILITY = (
    REPO_ROOT / "final" / "models" / "mlp_per_role" / "embeddings_raw_ally_utility.json"
)
DEFAULT_ARCHETYPES = REPO_ROOT / "final" / "data" / "champion_archetypes.json"
DEFAULT_CLASSES = REPO_ROOT / "final" / "data" / "champion_classes.json"
DEFAULT_TRAIN = REPO_ROOT / "final" / "data" / "training" / "train.parquet"
DEFAULT_OUTDIR = REPO_ROOT / "final" / "analysis" / "embedding_analysis"

SUPPORT_ARCHETYPE_ORDER = [
    "engage_tank",
    "hook",
    "roamer",
    "warden",
    "enchanter",
    "mage_support",
    "catcher",
]

SUPPORT_ARCHETYPE_COLORS = {
    "engage_tank": "#1f77b4",
    "hook": "#17becf",
    "roamer": "#d62728",
    "warden": "#9467bd",
    "enchanter": "#2ca02c",
    "mage_support": "#ff7f0e",
    "catcher": "#8c564b",
}

CLASS_COLORS = {
    "Assassin": "#d62728",
    "Fighter": "#ff7f0e",
    "Mage": "#9467bd",
    "Marksman": "#2ca02c",
    "Support": "#17becf",
    "Tank": "#1f77b4",
    "Unknown": "#7f7f7f",
}

NEAREST_NEIGHBOR_COLUMNS = [
    "neighbor_scope",
    "support_champion_id",
    "support_name",
    "support_archetype",
    "neighbor_rank",
    "neighbor_champion_id",
    "neighbor_name",
    "neighbor_support_archetype",
    "neighbor_primary_class",
    "neighbor_is_support",
    "cosine_distance",
    "cosine_similarity",
    "same_support_archetype",
    "same_primary_class",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze learned MLP champion embeddings.")
    parser.add_argument(
        "--embeddings",
        nargs="*",
        default=None,
        help=(
            "Embedding JSON files to analyze. Defaults to shared mlp_embed and "
            "per-role ally_utility embeddings when both exist."
        ),
    )
    parser.add_argument(
        "--embedding-labels",
        nargs="*",
        default=None,
        help="Optional labels matching --embeddings, used for output subdirectories.",
    )
    parser.add_argument("--champion-archetypes", default=str(DEFAULT_ARCHETYPES))
    parser.add_argument("--champion-classes", default=str(DEFAULT_CLASSES))
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--skip-projections",
        action="store_true",
        help="Skip t-SNE/UMAP figures; useful for smoke tests and metric-only runs.",
    )
    parser.add_argument("--annotate-supports", action="store_true", default=True)
    return parser.parse_args()


def sanitize_label(label: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in label.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "embedding"


def infer_embedding_label(path: Path) -> str:
    path_parts = {part.lower() for part in path.parts}
    stem = path.stem
    if stem == "embeddings_raw" and "mlp_embed" in path_parts:
        return "shared"
    if "mlp_per_role" in path_parts and stem.startswith("embeddings_raw_"):
        return sanitize_label(f"per_role_{stem[len('embeddings_raw_'):]}")
    return sanitize_label(stem)


def resolve_embedding_specs(args: argparse.Namespace) -> List[Tuple[str, Path]]:
    if args.embeddings:
        paths = [Path(path) for path in args.embeddings]
    else:
        paths = [
            path
            for path in [DEFAULT_SHARED_EMBEDDINGS, DEFAULT_PER_ROLE_ALLY_UTILITY]
            if path.exists()
        ]
        if not paths:
            paths = [DEFAULT_SHARED_EMBEDDINGS, DEFAULT_PER_ROLE_ALLY_UTILITY]

    if args.embedding_labels:
        if len(args.embedding_labels) != len(paths):
            raise ValueError("--embedding-labels must have the same length as --embeddings.")
        labels = [sanitize_label(label) for label in args.embedding_labels]
    else:
        labels = [infer_embedding_label(path) for path in paths]

    seen: Dict[str, int] = {}
    specs: List[Tuple[str, Path]] = []
    for label, path in zip(labels, paths):
        count = seen.get(label, 0)
        seen[label] = count + 1
        unique_label = label if count == 0 else f"{label}_{count + 1}"
        specs.append((unique_label, path))
    return specs


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_archetypes(path: Path) -> Dict[int, Dict[str, str]]:
    if not path.exists():
        print(f"[Warning] Archetype metadata not found at {path}; continuing without archetypes.")
        return {}
    raw = load_json(path)
    champions = raw.get("champions", {})
    out: Dict[int, Dict[str, str]] = {}
    for cid_str, info in champions.items():
        try:
            cid = int(cid_str)
        except ValueError:
            continue
        out[cid] = {
            "name": str(info.get("name", f"champ_{cid}")),
            "support_archetype": str(info.get("support", "")),
            "generic_archetype": str(info.get("generic", "")),
        }
    return out


def load_classes(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        print(f"[Warning] Class metadata not found at {path}; continuing without classes.")
        return {}
    raw = load_json(path)
    out: Dict[int, Dict[str, Any]] = {}
    for cid_str, info in raw.items():
        try:
            cid = int(cid_str)
        except ValueError:
            continue
        out[cid] = {
            "name": str(info.get("name", f"champ_{cid}")),
            "primary_class": str(info.get("primary_class", "Unknown")),
            "classes": list(info.get("classes", [])),
        }
    return out


def build_embedding_frame(
    embeddings_path: Path,
    archetypes: Dict[int, Dict[str, str]],
    classes: Dict[int, Dict[str, Any]],
) -> Tuple[pd.DataFrame, np.ndarray]:
    rows = load_json(embeddings_path)
    records: List[Dict[str, Any]] = []
    vectors: List[List[float]] = []

    for row in rows:
        cid = int(row["champion_id"])
        arch_info = archetypes.get(cid, {})
        class_info = classes.get(cid, {})
        support_arch = arch_info.get("support_archetype", "")
        primary_class = class_info.get("primary_class", "Unknown")
        champion_name = (
            class_info.get("name")
            or arch_info.get("name")
            or row.get("name")
            or f"champ_{cid}"
        )
        embedding = [float(v) for v in row["embedding"]]

        records.append(
            {
                "champion_id": cid,
                "name": champion_name,
                "vocab_index": int(row.get("vocab_index", -1)),
                "support_archetype": support_arch or "",
                "generic_archetype": arch_info.get("generic_archetype", row.get("archetype", "")),
                "is_support": bool(support_arch),
                "primary_class": primary_class,
                "classes": "|".join(class_info.get("classes", [])),
            }
        )
        vectors.append(embedding)

    df_unsorted = pd.DataFrame.from_records(records)
    sort_idx = df_unsorted.sort_values("vocab_index").index.to_numpy()
    df = df_unsorted.iloc[sort_idx].reset_index(drop=True)
    X = np.asarray(vectors, dtype=np.float32)[sort_idx]
    if len(df) != len(X):
        raise RuntimeError("Embedding metadata and vector matrix have different lengths.")
    return df, X


def run_tsne(X_norm: np.ndarray, perplexity: int, seed: int) -> np.ndarray:
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        metric="cosine",
        random_state=seed,
    ).fit_transform(X_norm)


def run_umap(X_norm: np.ndarray, n_neighbors: int, seed: int) -> np.ndarray:
    try:
        from umap import UMAP
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: umap-learn. Install it in the project venv with "
            "`.venv\\Scripts\\python.exe -m pip install umap-learn` and rerun this script."
        ) from exc

    return UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.15,
        metric="cosine",
        random_state=seed,
    ).fit_transform(X_norm)


def add_projection(df: pd.DataFrame, coords: np.ndarray, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out[f"{prefix}_x"] = coords[:, 0]
    out[f"{prefix}_y"] = coords[:, 1]
    return out


def plot_support_archetype(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    outpath: Path,
    title: str,
    annotate: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    non_support = ~df["is_support"]
    ax.scatter(
        df.loc[non_support, x_col],
        df.loc[non_support, y_col],
        s=24,
        color="#d0d4da",
        alpha=0.55,
        linewidths=0,
        label="non-support / no support archetype",
    )

    for arch in SUPPORT_ARCHETYPE_ORDER:
        subset = df[df["support_archetype"] == arch]
        if subset.empty:
            continue
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=76,
            color=SUPPORT_ARCHETYPE_COLORS.get(arch, "#333333"),
            alpha=0.88,
            edgecolors="white",
            linewidths=0.8,
            label=f"{arch} (n={len(subset)})",
        )
        if annotate:
            for _, row in subset.iterrows():
                ax.text(
                    row[x_col],
                    row[y_col],
                    str(row["name"]),
                    fontsize=7.2,
                    color="#222222",
                    ha="center",
                    va="bottom",
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.68},
                )

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")
    ax.grid(alpha=0.22)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_primary_class(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    outpath: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 7.0))
    classes = sorted(df["primary_class"].fillna("Unknown").unique())
    for cls in classes:
        subset = df[df["primary_class"] == cls]
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=46,
            color=CLASS_COLORS.get(cls, "#7f7f7f"),
            alpha=0.82,
            edgecolors="white",
            linewidths=0.45,
            label=f"{cls} (n={len(subset)})",
        )

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")
    ax.grid(alpha=0.22)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_neighbor_similarity(neighbors: pd.DataFrame, outpath: Path) -> None:
    if neighbors.empty:
        return

    support_order = (
        neighbors[neighbors["neighbor_rank"] == 1]
        .sort_values(["support_archetype", "support_name"])["support_name"]
        .tolist()
    )
    top = neighbors[neighbors["neighbor_rank"] <= 5].copy()
    top["support_name"] = pd.Categorical(top["support_name"], categories=support_order, ordered=True)
    top = top.sort_values(["support_name", "neighbor_rank"])

    pivot = top.pivot(index="support_name", columns="neighbor_rank", values="cosine_similarity")
    labels = top.pivot(index="support_name", columns="neighbor_rank", values="neighbor_name")

    fig_h = max(6.0, 0.26 * len(pivot))
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"#{int(c)}" for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=7.5)
    ax.set_title("Nearest support-only embedding neighbors for support champions", fontsize=13, weight="bold")
    ax.set_xlabel("neighbor rank")
    ax.set_ylabel("support champion")

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            name = labels.iloc[i, j]
            value = pivot.iloc[i, j]
            ax.text(
                j,
                i,
                str(name),
                ha="center",
                va="center",
                fontsize=6.2,
                color="white" if value < 0.55 else "#111111",
            )

    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def compute_roam_score_means(train_path: Path) -> Dict[int, float]:
    """Load training data and compute mean roam score per support champion."""
    df = pd.read_parquet(train_path, columns=["ally_utility_champion_id", "support_roam_score"])
    means = df.groupby("ally_utility_champion_id")["support_roam_score"].mean()
    return {int(cid): float(val) for cid, val in means.items()}


def embedding_distance_vs_roam_score(
    df: pd.DataFrame,
    X_norm: np.ndarray,
    roam_means: Dict[int, float],
    outdir: Path,
) -> Dict[str, Any]:
    """Correlate pairwise embedding distance with roam score difference."""
    from scipy.stats import spearmanr, pearsonr

    # Filter to expert support champions that have observed support-slot roam data.
    # This answers: among champions we consider supports, do embedding distances
    # track differences in their empirical roaming tendency?
    mask = (df["is_support"] & df["champion_id"].isin(roam_means.keys())).to_numpy()
    df_sub = df[mask].reset_index(drop=True)
    X_sub = X_norm[mask]

    if len(df_sub) < 5:
        return {"roam_correlation_n_pairs": 0}

    dist = cosine_distances(X_sub)
    n = len(df_sub)
    cids = df_sub["champion_id"].tolist()

    pair_rows = []
    pairs_dist = []
    pairs_roam_diff = []
    for i in range(n):
        for j in range(i + 1, n):
            rm_i = roam_means.get(cids[i])
            rm_j = roam_means.get(cids[j])
            if rm_i is not None and rm_j is not None:
                roam_diff = abs(rm_i - rm_j)
                emb_dist = float(dist[i, j])
                pairs_dist.append(emb_dist)
                pairs_roam_diff.append(roam_diff)
                pair_rows.append({
                    "champion_a_id": int(df_sub.loc[i, "champion_id"]),
                    "champion_a_name": df_sub.loc[i, "name"],
                    "champion_a_archetype": df_sub.loc[i, "support_archetype"],
                    "champion_a_mean_roam": float(rm_i),
                    "champion_b_id": int(df_sub.loc[j, "champion_id"]),
                    "champion_b_name": df_sub.loc[j, "name"],
                    "champion_b_archetype": df_sub.loc[j, "support_archetype"],
                    "champion_b_mean_roam": float(rm_j),
                    "cosine_distance": emb_dist,
                    "roam_score_abs_diff": float(roam_diff),
                    "same_support_archetype": (
                        df_sub.loc[i, "support_archetype"] == df_sub.loc[j, "support_archetype"]
                    ),
                })

    if len(pairs_dist) < 10:
        return {"roam_correlation_n_pairs": len(pairs_dist)}

    d = np.asarray(pairs_dist)
    r = np.asarray(pairs_roam_diff)
    pd.DataFrame(pair_rows).to_csv(outdir / "embedding_distance_vs_roam_pairs.csv", index=False)
    pearson_corr, pearson_p = pearsonr(d, r)
    spearman_corr, spearman_p = spearmanr(d, r)

    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(d, r, s=3, alpha=0.15, color="#2563eb", rasterized=True)

    # Add trend line
    z = np.polyfit(d, r, 1)
    x_line = np.linspace(d.min(), d.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), color="#dc2626", linewidth=2,
            label=f"Pearson r={pearson_corr:.3f} (p={pearson_p:.1e})")

    ax.set_xlabel("Cosine distance in embedding space", fontsize=11)
    ax.set_ylabel("|Mean roam score A - Mean roam score B|", fontsize=11)
    ax.set_title("Embedding distance vs. roam score difference\n(support champion pairs only)",
                 fontsize=12, weight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "embedding_distance_vs_roam.png", dpi=200)
    plt.close(fig)

    result = {
        "roam_correlation_n_pairs": len(pairs_dist),
        "roam_correlation_n_champions": n,
        "roam_pearson_r": float(pearson_corr),
        "roam_pearson_p": float(pearson_p),
        "roam_spearman_r": float(spearman_corr),
        "roam_spearman_p": float(spearman_p),
    }
    print(f"[Roam vs Embedding] n_pairs={len(pairs_dist):,}  "
          f"Pearson r={pearson_corr:.4f} (p={pearson_p:.2e})  "
          f"Spearman r={spearman_corr:.4f} (p={spearman_p:.2e})")
    return result


def plot_tsne_by_roam_score(
    df: pd.DataFrame,
    roam_means: Dict[int, float],
    x_col: str,
    y_col: str,
    outpath: Path,
    annotate_supports: bool,
) -> None:
    """t-SNE colored by mean roam score instead of archetype."""
    df = df.copy()
    df["mean_roam_score"] = df["champion_id"].map(roam_means)

    fig, ax = plt.subplots(figsize=(10.5, 7.2))

    # Champions without roam data (never played support)
    no_data = df["mean_roam_score"].isna()
    ax.scatter(df.loc[no_data, x_col], df.loc[no_data, y_col],
               s=20, color="#d0d4da", alpha=0.4, linewidths=0,
               label="no support data")

    # Champions with roam data - color by score
    has_data = ~no_data
    sc = ax.scatter(df.loc[has_data, x_col], df.loc[has_data, y_col],
                    c=df.loc[has_data, "mean_roam_score"],
                    s=60, cmap="RdYlGn", vmin=0.15, vmax=0.55,
                    alpha=0.85, edgecolors="white", linewidths=0.6)
    plt.colorbar(sc, ax=ax, label="Mean roam score", fraction=0.035, pad=0.02)

    if annotate_supports:
        supports = df[df["is_support"] & has_data]
        for _, row in supports.iterrows():
            ax.text(row[x_col], row[y_col], str(row["name"]),
                    fontsize=6.8, color="#222222", ha="center", va="bottom",
                    bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.65})

    ax.set_title("Champion embeddings colored by mean roam score (t-SNE, perplexity=30)",
                 fontsize=13, weight="bold")
    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def safe_silhouette(
    X_norm: np.ndarray,
    labels: Iterable[str],
    mask: Optional[np.ndarray] = None,
    min_group_size: int = 2,
) -> Tuple[float, int, int]:
    labels_arr = np.asarray(list(labels), dtype=object)
    if mask is None:
        mask = np.ones(len(labels_arr), dtype=bool)
    labels_arr = labels_arr[mask]
    X_sel = X_norm[mask]

    counts = pd.Series(labels_arr).value_counts()
    valid_labels = set(counts[counts >= min_group_size].index)
    valid_mask = np.asarray([label in valid_labels for label in labels_arr], dtype=bool)
    labels_arr = labels_arr[valid_mask]
    X_sel = X_sel[valid_mask]

    n_labels = len(set(labels_arr))
    if len(labels_arr) < 3 or n_labels < 2 or n_labels >= len(labels_arr):
        return float("nan"), int(len(labels_arr)), int(n_labels)

    score = float(silhouette_score(X_sel, labels_arr, metric="cosine"))
    return score, int(len(labels_arr)), int(n_labels)


def build_nearest_neighbors(df: pd.DataFrame, X_norm: np.ndarray, k: int = 5) -> pd.DataFrame:
    dist = cosine_distances(X_norm)
    records: List[Dict[str, Any]] = []
    support_indices = df.index[df["is_support"]].tolist()

    for support_idx in support_indices:
        ordered = np.argsort(dist[support_idx])
        ordered = [idx for idx in ordered if idx != support_idx][:k]
        support = df.loc[support_idx]
        for rank, neighbor_idx in enumerate(ordered, start=1):
            neighbor = df.loc[neighbor_idx]
            cosine_distance = float(dist[support_idx, neighbor_idx])
            records.append(
                {
                    "neighbor_scope": "all_champions_exploratory",
                    "support_champion_id": int(support["champion_id"]),
                    "support_name": support["name"],
                    "support_archetype": support["support_archetype"],
                    "neighbor_rank": rank,
                    "neighbor_champion_id": int(neighbor["champion_id"]),
                    "neighbor_name": neighbor["name"],
                    "neighbor_support_archetype": neighbor["support_archetype"],
                    "neighbor_primary_class": neighbor["primary_class"],
                    "neighbor_is_support": bool(neighbor["is_support"]),
                    "cosine_distance": cosine_distance,
                    "cosine_similarity": 1.0 - cosine_distance,
                    "same_support_archetype": (
                        bool(neighbor["is_support"])
                        and support["support_archetype"] == neighbor["support_archetype"]
                    ),
                    "same_primary_class": support["primary_class"] == neighbor["primary_class"],
                }
            )
    return pd.DataFrame.from_records(records, columns=NEAREST_NEIGHBOR_COLUMNS)


def build_nearest_neighbors_restricted(df: pd.DataFrame, X_norm: np.ndarray, k: int = 5) -> pd.DataFrame:
    dist = cosine_distances(X_norm)
    records: List[Dict[str, Any]] = []
    support_indices = df.index[df["is_support"]].tolist()

    for support_idx in support_indices:
        ordered = np.argsort(dist[support_idx])
        ordered_supports = [idx for idx in ordered if df.loc[idx, "is_support"] and idx != support_idx][:k]
        support = df.loc[support_idx]
        for rank, neighbor_idx in enumerate(ordered_supports, start=1):
            neighbor = df.loc[neighbor_idx]
            cosine_distance = float(dist[support_idx, neighbor_idx])
            records.append(
                {
                    "neighbor_scope": "support_only",
                    "support_champion_id": int(support["champion_id"]),
                    "support_name": support["name"],
                    "support_archetype": support["support_archetype"],
                    "neighbor_rank": rank,
                    "neighbor_champion_id": int(neighbor["champion_id"]),
                    "neighbor_name": neighbor["name"],
                    "neighbor_support_archetype": neighbor["support_archetype"],
                    "neighbor_primary_class": neighbor["primary_class"],
                    "neighbor_is_support": True,
                    "cosine_distance": cosine_distance,
                    "cosine_similarity": 1.0 - cosine_distance,
                    "same_support_archetype": support["support_archetype"] == neighbor["support_archetype"],
                    "same_primary_class": support["primary_class"] == neighbor["primary_class"],
                }
            )
    return pd.DataFrame.from_records(records, columns=NEAREST_NEIGHBOR_COLUMNS)


def safe_rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    values = frame[column].dropna()
    if values.empty:
        return float("nan")
    return float(values.astype(float).mean())


def compute_neighbor_metrics(
    all_neighbors: pd.DataFrame,
    support_only_neighbors: pd.DataFrame,
    top_k: int,
) -> Dict[str, Any]:
    all_rank1 = all_neighbors[all_neighbors["neighbor_rank"] == 1]
    support_rank1 = support_only_neighbors[support_only_neighbors["neighbor_rank"] == 1]

    return {
        "top_k": int(top_k),
        "all_champions_neighbor_scope": "exploratory_all_champions",
        "all_topk_n_rows": int(len(all_neighbors)),
        "all_rank1_n_rows": int(len(all_rank1)),
        "all_topk_support_neighbor_rate": safe_rate(all_neighbors, "neighbor_is_support"),
        "all_topk_same_support_archetype_rate": safe_rate(all_neighbors, "same_support_archetype"),
        "all_topk_same_primary_class_rate": safe_rate(all_neighbors, "same_primary_class"),
        "all_rank1_support_neighbor_rate": safe_rate(all_rank1, "neighbor_is_support"),
        "all_rank1_same_support_archetype_rate": safe_rate(all_rank1, "same_support_archetype"),
        "support_only_neighbor_scope": "main_support_only",
        "support_only_topk_n_rows": int(len(support_only_neighbors)),
        "support_only_rank1_n_rows": int(len(support_rank1)),
        "support_only_topk_same_support_archetype_rate": safe_rate(
            support_only_neighbors, "same_support_archetype"
        ),
        "support_only_topk_same_primary_class_rate": safe_rate(
            support_only_neighbors, "same_primary_class"
        ),
        "support_only_rank1_same_support_archetype_rate": safe_rate(
            support_rank1, "same_support_archetype"
        ),
    }


def format_float(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def write_summary(
    outpath: Path,
    df: pd.DataFrame,
    neighbor_df: pd.DataFrame,
    neighbor_restricted_df: pd.DataFrame,
    metrics: Dict[str, Any],
) -> None:
    support_df = df[df["is_support"]].copy()
    arch_counts = support_df["support_archetype"].value_counts().sort_index()
    class_counts = df["primary_class"].value_counts().sort_index()
    top_k = int(metrics.get("top_k", 5))

    def metric_rate(key: str) -> str:
        return format_float(float(metrics.get(key, float("nan"))))

    def neighbor_examples(frame: pd.DataFrame) -> List[str]:
        rank1 = frame[frame["neighbor_rank"] == 1].copy()
        if rank1.empty:
            return ["| n/a | n/a | n/a | n/a | n/a | n/a |"]
        rows = []
        for _, row in rank1.sort_values(["support_archetype", "support_name"]).iterrows():
            rows.append(
                f"| {row['support_name']} | {row['support_archetype'] or '-'} | "
                f"{row['neighbor_name']} | {row['neighbor_support_archetype'] or '-'} | "
                f"{row['neighbor_primary_class']} | {row['cosine_similarity']:.3f} |"
            )
        return rows

    lines = [
        "# Embedding analysis summary",
        "",
        "## Dataset",
        "",
        f"- Embedding label: `{metrics.get('embedding_label', 'embedding')}`",
        f"- Embedding file analyzed: `{metrics.get('embeddings_path', '')}`",
        f"- Champions with embeddings: {len(df)}",
        f"- Support-labeled champions: {len(support_df)}",
        f"- Embedding dimension: {metrics['embedding_dim']}",
        "",
        "Support archetype counts:",
        "",
    ]
    if arch_counts.empty:
        lines.append("- No support archetype metadata available.")
    else:
        lines.extend([f"- {arch}: {count}" for arch, count in arch_counts.items()])
    lines.extend(
        [
            "",
            "Primary Data Dragon class counts:",
            "",
        ]
    )
    lines.extend([f"- {cls}: {count}" for cls, count in class_counts.items()])
    lines.extend(
        [
            "",
            "## Cluster coherence",
            "",
            (
                f"- Support archetype silhouette, cosine metric: "
                f"{format_float(metrics['support_archetype_silhouette'])} "
                f"(n={metrics['support_archetype_silhouette_n']}, "
                f"groups={metrics['support_archetype_silhouette_groups']})"
            ),
            (
                f"- Primary class silhouette, cosine metric: "
                f"{format_float(metrics['primary_class_silhouette'])} "
                f"(n={metrics['primary_class_silhouette_n']}, "
                f"groups={metrics['primary_class_silhouette_groups']})"
            ),
            "",
            "Interpretation guide: silhouette near 1 implies compact, separated clusters; "
            "near 0 implies overlapping groups; negative values imply that many samples "
            "are closer to other groups than to their own.",
            "",
            "## Roam-score continuity",
            "",
        ]
    )
    if "roam_pearson_r" in metrics:
        lines.extend(
            [
                (
                    f"- Pairwise support-champion embedding distance vs. absolute mean roam-score "
                    f"difference: Pearson r={format_float(metrics['roam_pearson_r'])}, "
                    f"p={metrics['roam_pearson_p']:.2e}; "
                    f"Spearman rho={format_float(metrics['roam_spearman_r'])}, "
                    f"p={metrics['roam_spearman_p']:.2e}."
                ),
                (
                    f"- Pairs analyzed: {metrics['roam_correlation_n_pairs']} "
                    f"from {metrics['roam_correlation_n_champions']} support champions."
                ),
                "",
            ]
        )
    else:
        lines.extend(["- Roam-score continuity analysis skipped; no training means were available.", ""])
    lines.extend(
        [
            "## Nearest-neighbor structure",
            "",
            "### Neighbors among all champions (Exploratory / unrestricted)",
            (
                "These neighbors are useful for inspection, but they can mix support and "
                "non-support champions and should not be used as the main support-archetype check."
            ),
            f"- Top-{top_k} neighbors that are support-labeled: {metric_rate('all_topk_support_neighbor_rate')}",
            f"- Top-{top_k} neighbors with same support archetype: {metric_rate('all_topk_same_support_archetype_rate')}",
            f"- Top-{top_k} neighbors with same primary class: {metric_rate('all_topk_same_primary_class_rate')}",
            f"- Rank-1 nearest neighbor support-labeled rate: {metric_rate('all_rank1_support_neighbor_rate')}",
            f"- Rank-1 same support archetype rate: {metric_rate('all_rank1_same_support_archetype_rate')}",
            "",
            "### Neighbors restricted to support champions (Main support-only check)",
            f"- Top-{top_k} support neighbors with same support archetype: {metric_rate('support_only_topk_same_support_archetype_rate')}",
            f"- Top-{top_k} support neighbors with same primary class: {metric_rate('support_only_topk_same_primary_class_rate')}",
            f"- Rank-1 nearest support neighbor with same support archetype rate: {metric_rate('support_only_rank1_same_support_archetype_rate')}",
            "",
            "Rank-1 nearest neighbors for support champions (all champions, exploratory):",
            "",
            "| Support | Archetype | Nearest neighbor | Neighbor archetype | Neighbor class | Cosine sim. |",
            "|---|---|---|---|---|---:|",
        ]
    )
    lines.extend(neighbor_examples(neighbor_df))
    lines.extend(
        [
            "",
            "Rank-1 nearest neighbors restricted to support champions:",
            "",
            "| Support | Archetype | Nearest support neighbor | Neighbor archetype | Neighbor class | Cosine sim. |",
            "|---|---|---|---|---|---:|",
        ]
    )
    lines.extend(neighbor_examples(neighbor_restricted_df))
    lines.extend(
        [
            "",
            "## Figure outputs",
            "",
        ]
    )
    if metrics.get("projections_written"):
        lines.extend(
            [
                "- `tsne_by_archetype.png`: t-SNE, perplexity=30, support champions colored by support archetype.",
                "- `tsne_perplexity15_by_archetype.png`: t-SNE sensitivity check, perplexity=15.",
                "- `tsne_by_roam_score.png`: t-SNE colored by empirical mean support roam score.",
                "- `embedding_distance_vs_roam.png`: pairwise embedding distance vs. pairwise mean roam-score difference.",
                "- `umap_by_class.png`: UMAP, n_neighbors=15, all champions colored by primary Data Dragon class.",
                "- `umap_n10_by_class.png`: UMAP sensitivity check, n_neighbors=10.",
                "- `support_only_nearest_neighbors.png`: nearest-neighbor matrix restricted to support champions.",
            ]
        )
    else:
        lines.append("- Projection figures were skipped because `--skip-projections` was used.")
        if "roam_pearson_r" in metrics:
            lines.append("- `embedding_distance_vs_roam.png`: pairwise embedding distance vs. pairwise mean roam-score difference.")
        if len(neighbor_restricted_df):
            lines.append("- `support_only_nearest_neighbors.png`: nearest-neighbor matrix restricted to support champions.")
    lines.extend(["", "## Suggested wording for the report", ""])

    support_same_rate = float(metrics.get("support_only_topk_same_support_archetype_rate", float("nan")))
    support_silhouette = float(metrics.get("support_archetype_silhouette", float("nan")))
    if "roam_pearson_r" in metrics and metrics["roam_pearson_r"] > 0 and metrics["roam_pearson_p"] < 0.05:
        lines.extend(
            [
                "The embedding space shows measurable continuity with empirical roaming tendency: "
                "support-pair embedding distance increases with pairwise mean roam-score difference. "
                "This supports a cautious task-specific interpretation, but it is not by itself "
                "evidence of a general semantic champion map.",
            ]
        )
    elif (
        not math.isnan(support_silhouette)
        and support_silhouette > 0.08
        and not math.isnan(support_same_rate)
        and support_same_rate > 0.35
    ):
        lines.extend(
            [
                "The embedding space shows weak archetype-aligned neighborhood structure among "
                "support champions. This can be reported as limited evidence that the model learned "
                "task-relevant champion-level tendencies, not as proof that the embeddings are "
                "semantically organized.",
            ]
        )
    else:
        lines.extend(
            [
                "These checks do not support a strong semantic-embedding claim. The embeddings may "
                "still encode task-specific signal for prediction, but the observed support-only "
                "neighborhood and archetype-alignment metrics are not enough to describe the space "
                "as a clean semantic map.",
            ]
        )

    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_embedding_set(
    label: str,
    embeddings_path: Path,
    args: argparse.Namespace,
    archetypes: Dict[int, Dict[str, str]],
    classes: Dict[int, Dict[str, Any]],
    roam_means: Dict[int, float],
    output_root: Path,
    use_label_subdir: bool,
) -> Dict[str, Any]:
    outdir = output_root / label if use_label_subdir else output_root
    outdir.mkdir(parents=True, exist_ok=True)
    top_k = max(1, int(args.top_k))

    print(f"[Analyze] label={label} embeddings={embeddings_path}")
    df, X = build_embedding_frame(embeddings_path, archetypes, classes)
    X_norm = normalize(X, norm="l2")
    df.to_csv(outdir / "embedding_champions.csv", index=False)
    print(f"[Data:{label}] champions={len(df)}  dim={X.shape[1]}  supports={int(df['is_support'].sum())}")

    projections_written = False
    if args.skip_projections:
        print(f"[Projection:{label}] skipped by --skip-projections")
    else:
        print(f"[t-SNE:{label}] perplexity=15")
        tsne15 = run_tsne(X_norm, perplexity=15, seed=args.seed)
        print(f"[t-SNE:{label}] perplexity=30")
        tsne30 = run_tsne(X_norm, perplexity=30, seed=args.seed)
        print(f"[UMAP:{label}] n_neighbors=10")
        umap10 = run_umap(X_norm, n_neighbors=10, seed=args.seed)
        print(f"[UMAP:{label}] n_neighbors=15")
        umap15 = run_umap(X_norm, n_neighbors=15, seed=args.seed)

        projections = df.copy()
        for prefix, coords in [
            ("tsne_p15", tsne15),
            ("tsne_p30", tsne30),
            ("umap_n10", umap10),
            ("umap_n15", umap15),
        ]:
            projections = add_projection(projections, coords, prefix)
        projections.to_csv(outdir / "embedding_projections.csv", index=False)

        plot_support_archetype(
            projections,
            "tsne_p30_x",
            "tsne_p30_y",
            outdir / "tsne_by_archetype.png",
            f"{label}: champion embeddings by support archetype (t-SNE, perplexity=30)",
            annotate=args.annotate_supports,
        )
        plot_support_archetype(
            projections,
            "tsne_p15_x",
            "tsne_p15_y",
            outdir / "tsne_perplexity15_by_archetype.png",
            f"{label}: champion embeddings by support archetype (t-SNE, perplexity=15)",
            annotate=args.annotate_supports,
        )
        plot_primary_class(
            projections,
            "umap_n15_x",
            "umap_n15_y",
            outdir / "umap_by_class.png",
            f"{label}: champion embeddings by Data Dragon primary class (UMAP, n_neighbors=15)",
        )
        plot_primary_class(
            projections,
            "umap_n10_x",
            "umap_n10_y",
            outdir / "umap_n10_by_class.png",
            f"{label}: champion embeddings by Data Dragon primary class (UMAP, n_neighbors=10)",
        )
        if roam_means:
            plot_tsne_by_roam_score(
                projections,
                roam_means,
                "tsne_p30_x",
                "tsne_p30_y",
                outdir / "tsne_by_roam_score.png",
                annotate_supports=args.annotate_supports,
            )
        projections_written = True

    support_mask = df["is_support"].to_numpy(dtype=bool)
    support_sil, support_sil_n, support_sil_groups = safe_silhouette(
        X_norm,
        df["support_archetype"].fillna(""),
        mask=support_mask,
    )
    class_sil, class_sil_n, class_sil_groups = safe_silhouette(
        X_norm,
        df["primary_class"].fillna("Unknown"),
        mask=None,
    )

    nearest_all = build_nearest_neighbors(df, X_norm, k=top_k)
    nearest_all.to_csv(outdir / "nearest_neighbors_all_champions.csv", index=False)
    nearest_all.to_csv(outdir / "nearest_neighbors.csv", index=False)

    nearest_support = build_nearest_neighbors_restricted(df, X_norm, k=top_k)
    nearest_support.to_csv(outdir / "nearest_neighbors_support_only.csv", index=False)
    nearest_support.to_csv(outdir / "nearest_neighbors_restricted_supports.csv", index=False)
    plot_neighbor_similarity(nearest_support, outdir / "support_only_nearest_neighbors.png")
    plot_neighbor_similarity(nearest_support, outdir / "support_nearest_neighbors.png")

    roam_corr_metrics: Dict[str, Any] = {}
    if roam_means:
        roam_corr_metrics = embedding_distance_vs_roam_score(
            df,
            X_norm,
            roam_means,
            outdir,
        )

    neighbor_metrics = compute_neighbor_metrics(nearest_all, nearest_support, top_k=top_k)
    metrics = {
        "embedding_label": label,
        "embeddings_path": str(embeddings_path.resolve()),
        "output_dir": str(outdir.resolve()),
        "n_champions": int(len(df)),
        "n_support_labeled_champions": int(df["is_support"].sum()),
        "embedding_dim": int(X.shape[1]),
        "normalization": "l2",
        "distance_metric": "cosine",
        "tsne_perplexities": [15, 30] if projections_written else [],
        "umap_n_neighbors": [10, 15] if projections_written else [],
        "projections_written": bool(projections_written),
        "support_archetype_silhouette": support_sil,
        "support_archetype_silhouette_n": support_sil_n,
        "support_archetype_silhouette_groups": support_sil_groups,
        "primary_class_silhouette": class_sil,
        "primary_class_silhouette_n": class_sil_n,
        "primary_class_silhouette_groups": class_sil_groups,
        **neighbor_metrics,
        **roam_corr_metrics,
    }
    if top_k == 5:
        metrics.update(
            {
                "top5_support_neighbor_rate": metrics["all_topk_support_neighbor_rate"],
                "top5_same_support_archetype_rate": metrics["all_topk_same_support_archetype_rate"],
                "top5_same_primary_class_rate": metrics["all_topk_same_primary_class_rate"],
                "restricted_top5_same_support_archetype_rate": metrics[
                    "support_only_topk_same_support_archetype_rate"
                ],
                "restricted_top5_same_primary_class_rate": metrics[
                    "support_only_topk_same_primary_class_rate"
                ],
                "restricted_rank1_same_support_archetype_rate": metrics[
                    "support_only_rank1_same_support_archetype_rate"
                ],
            }
        )

    (outdir / "embedding_analysis_metadata.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_summary(outdir / "embedding_analysis_summary.md", df, nearest_all, nearest_support, metrics)

    print(f"[Metrics:{label}] support silhouette={format_float(support_sil)}")
    print(
        f"[Metrics:{label}] support-only top-{top_k} same archetype="
        f"{format_float(metrics['support_only_topk_same_support_archetype_rate'])}"
    )
    if "roam_pearson_r" in roam_corr_metrics:
        print(
            f"[Metrics:{label}] embedding dist vs roam diff Pearson="
            f"{format_float(roam_corr_metrics['roam_pearson_r'])}"
        )
    print(f"[Saved:{label}] {outdir.resolve()}")
    return metrics


def write_comparison(outdir: Path, metrics_rows: List[Dict[str, Any]]) -> None:
    comparison = pd.DataFrame(metrics_rows)
    comparison.to_csv(outdir / "embedding_analysis_comparison.csv", index=False)
    (outdir / "embedding_analysis_comparison.json").write_text(
        json.dumps(metrics_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Embedding analysis comparison",
        "",
        "Each row uses the same metadata and train-derived roam means, but a different embedding JSON.",
        "",
        "| Embedding | File | Support-only top-k same archetype | Support-only rank-1 same archetype | Roam Pearson r | Roam Spearman rho |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        lines.append(
            f"| {row['embedding_label']} | `{row['embeddings_path']}` | "
            f"{format_float(row.get('support_only_topk_same_support_archetype_rate', float('nan')))} | "
            f"{format_float(row.get('support_only_rank1_same_support_archetype_rate', float('nan')))} | "
            f"{format_float(row.get('roam_pearson_r', float('nan')))} | "
            f"{format_float(row.get('roam_spearman_r', float('nan')))} |"
        )
    (outdir / "embedding_analysis_comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    embedding_specs = resolve_embedding_specs(args)
    archetypes_path = Path(args.champion_archetypes)
    classes_path = Path(args.champion_classes)
    train_path = Path(args.train_data)

    print("[Embedding files]")
    for label, path in embedding_specs:
        print(f"  - {label}: {path}")

    archetypes = load_archetypes(archetypes_path)
    classes = load_classes(classes_path)

    roam_means: Dict[int, float] = {}
    if train_path.exists():
        roam_means = compute_roam_score_means(train_path)
        print(f"[Roam] Loaded mean roam scores for {len(roam_means)} champions from train only")
    else:
        print(f"[Roam] Train data not found at {train_path}, skipping roam analysis")

    use_label_subdir = len(embedding_specs) > 1
    metrics_rows = [
        analyze_embedding_set(
            label=label,
            embeddings_path=path,
            args=args,
            archetypes=archetypes,
            classes=classes,
            roam_means=roam_means,
            output_root=outdir,
            use_label_subdir=use_label_subdir,
        )
        for label, path in embedding_specs
    ]
    write_comparison(outdir, metrics_rows)
    print(f"[Saved] comparison={outdir.resolve()}")


if __name__ == "__main__":
    main()
