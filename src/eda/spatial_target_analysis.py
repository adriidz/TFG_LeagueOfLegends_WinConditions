import argparse
import csv
import json
import math
import os
import time
from collections import Counter
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

import sys
import os

# Importamos toda la geometría centralizada del nuevo pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02_data_processing"))
from shared_utils import (
    BLUE_TEAM_ID, RED_TEAM_ID, CANONICAL_ROLES, ROLE_KEYS_LOWER,
    DEFAULT_MIN_DURATION_MINUTES, DEFAULT_MAX_MINUTE, MAP_MAX, MAP_CENTER_SUM,
    EMPTY_ROLE_VALUES,
    BLUE_BASE_POLYGON, RED_BASE_POLYGON,
    TOP_LANE_CENTERLINE, BOT_LANE_CENTERLINE, MID_LANE_CENTERLINE, RIVER_CENTERLINE,
    BARON_PIT_CENTER, DRAGON_PIT_CENTER, PIT_RADIUS,
    get_team_relative_zone, classify_map_zone, get_participant_frame, extract_position,
    participant_is_alive, extract_team_role_map
)

DEFAULT_RAW_ROOT = os.path.join("data/raw", "raw")
DEFAULT_REGION = "europe"
DEFAULT_OUT_DIR = os.path.join("data/clean", "geometry_reports")
MAP_MIN = 0.0
# MAP_MAX imported from shared_utils (14800)

BALANCED_HEATMAP_PERCENTILE = 0.99
BALANCED_BORDER_MARGIN = 700.0
BALANCED_CORNER_MASK = 1600.0

ZONE_LABELS = [
    "OWN_BASE",
    "ENEMY_BASE",
    "TOP_LANE",
    "MID_LANE",
    "BOTTOM_LANE",
    "RIVER_TOP",
    "RIVER_BOT",
    "OWN_TOP_JUNGLE",
    "OWN_BOTTOM_JUNGLE",
    "ENEMY_TOP_JUNGLE",
    "ENEMY_BOTTOM_JUNGLE",
]

LANE_OR_RIVER_ZONES = {"TOP_LANE", "MID_LANE", "BOTTOM_LANE", "RIVER_TOP", "RIVER_BOT"}
OWN_JUNGLE_ZONES = {"OWN_TOP_JUNGLE", "OWN_BOTTOM_JUNGLE"}

ZONE_COLOR_MAP = {
    "OWN_BASE": "#1f77b4",
    "ENEMY_BASE": "#0b3a77",
    "TOP_LANE": "#d62728",
    "MID_LANE": "#ffbf00",
    "BOTTOM_LANE": "#2ca02c",
    "RIVER_TOP": "#17becf",
    "RIVER_BOT": "#17becf",
    "OWN_TOP_JUNGLE": "#9467bd",
    "OWN_BOTTOM_JUNGLE": "#8c564b",
    "ENEMY_TOP_JUNGLE": "#c5b0d5",
    "ENEMY_BOTTOM_JUNGLE": "#c49c94",
}



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def apply_sample_suffix(path: str, frac: Optional[float]) -> str:
    if frac is None or frac >= 1.0 or frac <= 0.0:
        return path
    base, ext = os.path.splitext(path)
    if ext == "":
        return f"{base}_sample{int(frac * 100)}"
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


def list_match_dirs(base: str) -> List[str]:
    if not os.path.isdir(base):
        raise SystemExit(f"No existe el directorio RAW: {base}")
    out: List[str] = []
    for name in os.listdir(base):
        mdir = os.path.join(base, name)
        if os.path.isdir(mdir):
            out.append(mdir)
    return out


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_match_info(match: dict) -> dict:
    return (match or {}).get("info") or {}


def get_timeline_frames(timeline: dict) -> List[dict]:
    info = (timeline or {}).get("info")
    if isinstance(info, dict):
        frames = info.get("frames")
        if isinstance(frames, list):
            return frames
    frames = (timeline or {}).get("frames")
    return frames if isinstance(frames, list) else []


def game_duration_minutes(info: dict) -> Optional[float]:
    raw = info.get("gameDuration")
    if raw is not None:
        try:
            value = float(raw)
            if value > 100000:
                value /= 1000.0
            return value / 60.0
        except Exception:
            pass

    game_creation = info.get("gameCreation")
    game_end = info.get("gameEndTimestamp")
    try:
        if game_creation is not None and game_end is not None:
            return max(0.0, (float(game_end) - float(game_creation)) / 1000.0 / 60.0)
    except Exception:
        pass
    return None


def normalize_role(value: object) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip().upper()
    if value in EMPTY_ROLE_VALUES:
        return None
    return value


def extract_team_role_map(info: dict) -> Dict[int, Dict[str, int]]:
    participants = list(info.get("participants") or [])
    team_role_map: Dict[int, Dict[str, int]] = {}

    for team_id in (BLUE_TEAM_ID, RED_TEAM_ID):
        team_participants = [p for p in participants if p.get("teamId") == team_id]
        role_to_pid: Dict[str, List[int]] = {role: [] for role in CANONICAL_ROLES}
        for p in team_participants:
            role = normalize_role(p.get("teamPosition"))
            pid = p.get("participantId")
            if role in role_to_pid and isinstance(pid, int):
                role_to_pid[role].append(pid)

        if all(len(role_to_pid[role]) == 1 for role in CANONICAL_ROLES):
            team_role_map[team_id] = {role: role_to_pid[role][0] for role in CANONICAL_ROLES}

    return team_role_map


def has_perfect_roles(info: dict) -> bool:
    role_map = extract_team_role_map(info)
    return BLUE_TEAM_ID in role_map and RED_TEAM_ID in role_map


def get_participant_frame(frame: dict, participant_id: int) -> Optional[dict]:
    participant_frames = frame.get("participantFrames") or {}
    if isinstance(participant_frames, dict):
        return participant_frames.get(str(participant_id)) or participant_frames.get(participant_id)
    return None


def extract_position_from_pf(participant_frame: Optional[dict]) -> Optional[Tuple[float, float]]:
    if not isinstance(participant_frame, dict):
        return None
    pos = participant_frame.get("position") or {}
    x = pos.get("x")
    y = pos.get("y")
    try:
        if x is None or y is None:
            return None
        return float(x), float(y)
    except Exception:
        return None


def participant_is_alive(participant_frame: Optional[dict]) -> bool:
    if not isinstance(participant_frame, dict):
        return False

    for key in ("isAlive", "alive"):
        if key in participant_frame:
            try:
                return bool(participant_frame[key])
            except Exception:
                pass

    champion_stats = participant_frame.get("championStats") or {}
    for hp_key in ("currentHealth", "health"):
        if hp_key in champion_stats:
            try:
                return float(champion_stats[hp_key]) > 0.0
            except Exception:
                pass

    pos = participant_frame.get("position") or {}
    x = pos.get("x")
    y = pos.get("y")
    try:
        if x is None or y is None:
            return False
        x = float(x)
        y = float(y)
    except Exception:
        return False

    return not (x <= 1.0 and y <= 1.0)


def get_frame_timestamp_ms(frame: dict, frame_index: int) -> float:
    ts = frame.get("timestamp")
    try:
        if ts is not None:
            return float(ts)
    except Exception:
        pass
    return float(frame_index) * 60_000.0


# ---------------------------------------------------------------------------
# Funciones importadas de shared_utils.py sustituyen a la lógica antigua
# ---------------------------------------------------------------------------



def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def mirror_to_blue_side(pos: Tuple[float, float], team_id: int, mirror_red: bool) -> Tuple[float, float]:
    x, y = pos
    if mirror_red and team_id == RED_TEAM_ID:
        return (MAP_MAX - x, MAP_MAX - y)
    return (x, y)


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * p))))
    return ordered[idx]


def fmt_float(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def print_header(title: str) -> None:
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))


def frames_upto_minute(frames: Iterable[dict], max_minute: float) -> List[dict]:
    max_ts = max_minute * 60.0 * 1000.0
    kept: List[dict] = []
    for frame in frames:
        ts = frame.get("timestamp")
        try:
            ts_value = float(ts) if ts is not None else None
        except Exception:
            ts_value = None
        if ts_value is None or ts_value <= max_ts:
            kept.append(frame)
    return kept


def keep_point_for_balanced_heatmap(
    x: float,
    y: float,
    border_margin: float = BALANCED_BORDER_MARGIN,
    corner_mask: float = BALANCED_CORNER_MASK,
) -> bool:
    if x < border_margin or y < border_margin:
        return False
    if x > MAP_MAX - border_margin or y > MAP_MAX - border_margin:
        return False
    if x <= corner_mask and y <= corner_mask:
        return False
    if x >= MAP_MAX - corner_mask and y >= MAP_MAX - corner_mask:
        return False
    return True


def filter_points_for_balanced_heatmap(
    xs: List[float],
    ys: List[float],
    border_margin: float = BALANCED_BORDER_MARGIN,
    corner_mask: float = BALANCED_CORNER_MASK,
) -> Tuple[List[float], List[float]]:
    out_xs: List[float] = []
    out_ys: List[float] = []
    for x, y in zip(xs, ys):
        if keep_point_for_balanced_heatmap(x, y, border_margin=border_margin, corner_mask=corner_mask):
            out_xs.append(x)
            out_ys.append(y)
    return out_xs, out_ys


def draw_polyline(ax, polyline: Sequence[Tuple[float, float]], color: str, label: Optional[str] = None) -> None:
    xs = [p[0] for p in polyline]
    ys = [p[1] for p in polyline]
    ax.plot(xs, ys, color=color, lw=2.0, linestyle="--", label=label)


def draw_spatial_boundaries(ax) -> None:
    ax.add_patch(patches.Polygon(BLUE_BASE_POLYGON, fill=False, edgecolor="blue", lw=2, label="Bases"))
    ax.add_patch(patches.Polygon(RED_BASE_POLYGON, fill=False, edgecolor="blue", lw=2))

    draw_polyline(ax, TOP_LANE_CENTERLINE, color="magenta", label="Side Lanes")
    draw_polyline(ax, BOT_LANE_CENTERLINE, color="magenta")
    draw_polyline(ax, MID_LANE_CENTERLINE, color="yellow", label="Mid Lane")
    draw_polyline(ax, RIVER_CENTERLINE, color="cyan", label="Río")

    ax.add_patch(patches.Circle(BARON_PIT_CENTER, PIT_RADIUS, fill=False, edgecolor="cyan", lw=1.8, linestyle=":"))
    ax.add_patch(patches.Circle(DRAGON_PIT_CENTER, PIT_RADIUS, fill=False, edgecolor="cyan", lw=1.8, linestyle=":"))
    # ax.add_patch(patches.Polygon(BARON_RIVER_POLYGON, fill=False, edgecolor="cyan", lw=1.5, linestyle="-."))
    # ax.add_patch(patches.Polygon(DRAGON_RIVER_POLYGON, fill=False, edgecolor="cyan", lw=1.5, linestyle="-."))

    ax.legend(loc="upper right", fontsize="small", framealpha=0.9)


def plot_hexbin_heatmap(
    xs: List[float],
    ys: List[float],
    out_path: str,
    title: str,
    map_image_path: Optional[str] = None,
    balanced: bool = False,
    color_cap_percentile: Optional[float] = None,
) -> None:
    if not xs or not ys:
        return

    plot_xs = xs
    plot_ys = ys

    if balanced:
        plot_xs, plot_ys = filter_points_for_balanced_heatmap(xs, ys)

    if not plot_xs or not plot_ys:
        return

    plt.figure(figsize=(8, 8))

    if map_image_path and os.path.exists(map_image_path):
        img = plt.imread(map_image_path)
        plt.imshow(img, extent=(MAP_MIN, MAP_MAX, MAP_MIN, MAP_MAX), origin="lower")
        alpha = 0.35
    else:
        alpha = 0.6

    hb = plt.hexbin(
        plot_xs,
        plot_ys,
        gridsize=55,
        extent=(MAP_MIN, MAP_MAX, MAP_MIN, MAP_MAX),
        mincnt=1,
        alpha=alpha,
        norm=mcolors.LogNorm(vmin=1.0),
        zorder=2,
    )

    if color_cap_percentile is not None:
        counts = [float(v) for v in hb.get_array() if float(v) > 0.0]
        vmax = percentile(counts, color_cap_percentile)
        if vmax is not None and vmax > 1.0:
            hb.set_clim(1.0, vmax)

    draw_spatial_boundaries(plt.gca())
    plt.xlim(MAP_MIN, MAP_MAX)
    plt.ylim(MAP_MIN, MAP_MAX)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    cb = plt.colorbar(hb)
    cb.set_label("Densidad (escala log)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def build_zone_reference_grid(grid_size: int = 320) -> Tuple[np.ndarray, mcolors.ListedColormap]:
    xs = np.linspace(MAP_MIN, MAP_MAX, grid_size)
    ys = np.linspace(MAP_MIN, MAP_MAX, grid_size)
    zone_to_id = {zone: idx for idx, zone in enumerate(ZONE_LABELS)}
    img = np.zeros((grid_size, grid_size), dtype=np.int32)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            zone = get_team_relative_zone(float(x), float(y), BLUE_TEAM_ID)
            img[iy, ix] = zone_to_id[zone]

    cmap = mcolors.ListedColormap([ZONE_COLOR_MAP[z] for z in ZONE_LABELS])
    return img, cmap


def draw_zone_reference_overlay(
    ax,
    grid_size: int = 320,
    alpha: float = 0.5,
    interpolation: str = "nearest",
) -> None:
    img, cmap = build_zone_reference_grid(grid_size=grid_size)
    ax.imshow(
        img,
        origin="lower",
        extent=(MAP_MIN, MAP_MAX, MAP_MIN, MAP_MAX),
        interpolation=interpolation,
        cmap=cmap,
        alpha=alpha,
        vmin=0,
        vmax=len(ZONE_LABELS) - 1,
        zorder=3,
    )


def plot_zone_reference(
    out_path: str,
    map_image_path: Optional[str] = None,
    grid_size: int = 320,
    background_xs: Optional[List[float]] = None,
    background_ys: Optional[List[float]] = None,
    balanced_background: bool = False,
    heatmap_gridsize: int = 55,
    heatmap_alpha: float = 0.60,
    zone_alpha: float = 0.26,
) -> None:
    plt.figure(figsize=(8.4, 8.2))
    ax = plt.gca()

    if map_image_path and os.path.exists(map_image_path):
        base_img = plt.imread(map_image_path)
        ax.imshow(base_img, extent=(MAP_MIN, MAP_MAX, MAP_MIN, MAP_MAX), origin="lower", alpha=0.25, zorder=0)

    hb = None
    if background_xs and background_ys:
        plot_xs = background_xs
        plot_ys = background_ys
        if balanced_background:
            plot_xs, plot_ys = filter_points_for_balanced_heatmap(background_xs, background_ys)
        if plot_xs and plot_ys:
            hb = ax.hexbin(
                plot_xs,
                plot_ys,
                gridsize=heatmap_gridsize,
                extent=(MAP_MIN, MAP_MAX, MAP_MIN, MAP_MAX),
                mincnt=1,
                alpha=heatmap_alpha,
                norm=mcolors.LogNorm(vmin=1.0),
                zorder=2,
            )

    draw_zone_reference_overlay(ax, grid_size=grid_size, alpha=zone_alpha)
    draw_spatial_boundaries(ax)
    ax.set_xlim(MAP_MIN, MAP_MAX)
    ax.set_ylim(MAP_MIN, MAP_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if hb is not None:
        ax.set_title("Referencia geométrica sobre heatmap real")
        cb = plt.colorbar(hb)
        cb.set_label("Densidad del heatmap (escala log)")
    else:
        ax.set_title("Referencia geométrica canónica de zonas")

    handles = [
        patches.Patch(facecolor=ZONE_COLOR_MAP[zone], edgecolor="none", alpha=0.75, label=zone)
        for zone in ZONE_LABELS
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_histogram(
    values: List[float],
    out_path: str,
    title: str,
    xlabel: str,
    bins: int = 40,
    xlim: Optional[Tuple[float, float]] = None,
    vertical_lines: Optional[List[Tuple[float, str]]] = None,
) -> None:
    if not values:
        return

    plt.figure(figsize=(9, 6))
    plt.hist(values, bins=bins)
    if xlim is not None:
        plt.xlim(*xlim)
    if vertical_lines:
        for x, label in vertical_lines:
            plt.axvline(x, linestyle="--", linewidth=2, label=label)
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Número de observaciones")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_counter(counter: Counter, out_path: str, title: str, xlabel: str, ylabel: str) -> None:
    if not counter:
        return
    labels = list(counter.keys())
    values = [counter[k] for k in labels]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def summarize_distribution(title: str, values: List[float], unit: str = "") -> None:
    print_header(title)
    if not values:
        print("Sin observaciones válidas.")
        return

    print(f"n = {len(values)}")
    print(f"media   = {fmt_float(mean(values))}{unit}")
    print(f"mediana = {fmt_float(median(values))}{unit}")
    print(f"p10     = {fmt_float(percentile(values, 0.10))}{unit}")
    print(f"p25     = {fmt_float(percentile(values, 0.25))}{unit}")
    print(f"p75     = {fmt_float(percentile(values, 0.75))}{unit}")
    print(f"p90     = {fmt_float(percentile(values, 0.90))}{unit}")
    print(f"mín     = {fmt_float(min(values))}{unit}")
    print(f"máx     = {fmt_float(max(values))}{unit}")


def write_zone_rows_csv(rows: List[Dict[str, object]], out_path: str) -> None:
    if not rows:
        return
    fieldnames = [
        "match_id",
        "team_id",
        "participant_id",
        "timestamp_ms",
        "minute",
        "x",
        "y",
        "canonical_x",
        "canonical_y",
        "zone",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Análisis espacial early game desde timeline.json (0-14 min)")
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--map-image", default=None, help="PNG/JPG opcional del mapa para superponer los heatmaps")
    parser.add_argument("--min-duration-minutes", type=float, default=15.0)
    parser.add_argument("--max-minute", type=float, default=DEFAULT_MAX_MINUTE)
    parser.add_argument("--require-perfect-roles", action="store_true")
    parser.add_argument("--mirror-red-to-blue", action="store_true", default=True)
    parser.add_argument("--allow-original-orientation", dest="mirror_red_to_blue", action="store_false")
    parser.add_argument("--min-shared-frames", type=int, default=8, help="Mínimo de frames con Support y ADC posicionados para computar su media")
    parser.add_argument("--min-jungle-classified-frames", type=int, default=8, help="Mínimo de frames clasificables para computar ratio del jungla")
    parser.add_argument("--max-matches", type=int, default=50000, help="Límite de partidas a procesar para pruebas rápidas")
    parser.add_argument("--export-jungle-zones-csv", action="store_true", help="Exporta un CSV con la zona del jungla frame a frame")
    parser.add_argument("--sample-frac", type=float, default=None, help="Fracción (ej 0.1) o lee TFG_SAMPLE_FRAC.")
    args = parser.parse_args()

    target_frac = get_target_frac(args.sample_frac)
    if target_frac is not None and 0.0 < target_frac < 1.0:
        args.out_dir = apply_sample_suffix(args.out_dir, target_frac)
        print(f"Muestreo de EDA detectado ({target_frac}). Reportes irán a {args.out_dir}")

    ensure_dir(args.out_dir)
    base = os.path.join(args.raw_root, args.region)
    match_dirs = list_match_dirs(base)
    
    if target_frac is not None and 0.0 < target_frac < 1.0:
        limit = max(1, int(len(match_dirs) * target_frac))
        match_dirs = match_dirs[:limit]
        print(f"Muestreo aplicado ({target_frac}): Limitado a {limit} partidas para EDA.")
        
    if args.max_matches and args.max_matches > 0:
        match_dirs = match_dirs[:args.max_matches]

    total_seen = 0
    total_kept = 0
    bad_match_json = 0
    bad_timeline_json = 0
    missing_info = 0
    filtered_short = 0
    filtered_roles = 0
    no_frames = 0
    incomplete_role_map = 0

    progress_every = 1000
    t0 = time.time()
    last_log = t0

    jungle_xs: List[float] = []
    jungle_ys: List[float] = []
    support_xs: List[float] = []
    support_ys: List[float] = []
    support_adc_mean_distances: List[float] = []
    jungle_presence_ratios: List[float] = []
    jungle_zone_counter: Counter = Counter()
    jungle_zone_rows: List[Dict[str, object]] = []

    duo_team_observations = 0
    jungle_team_observations = 0
    sample_bad_matches: List[str] = []

    for mdir in match_dirs:
        total_seen += 1

        if total_seen % progress_every == 0 or (time.time() - last_log) > 15:
            now = time.time()
            elapsed = now - t0
            rate = total_seen / elapsed if elapsed > 0 else 0.0
            print(
                f"[{total_seen}/{len(match_dirs)}] "
                f"kept={total_kept} "
                f"bad_match={bad_match_json} "
                f"bad_tl={bad_timeline_json} "
                f"short={filtered_short} "
                f"roles={filtered_roles + incomplete_role_map} "
                f"noframes={no_frames} "
                f"rate={rate:.1f} matches/s"
            )
            last_log = now

        match_path = os.path.join(mdir, "match.json")
        timeline_path = os.path.join(mdir, "timeline.json")

        try:
            match = load_json(match_path)
        except Exception:
            bad_match_json += 1
            if len(sample_bad_matches) < 8:
                sample_bad_matches.append(os.path.basename(mdir))
            continue

        try:
            timeline = load_json(timeline_path)
        except Exception:
            bad_timeline_json += 1
            if len(sample_bad_matches) < 8:
                sample_bad_matches.append(os.path.basename(mdir))
            continue

        info = get_match_info(match)
        if not info:
            missing_info += 1
            continue

        dur_min = game_duration_minutes(info)
        if dur_min is None or dur_min < args.min_duration_minutes:
            filtered_short += 1
            continue

        role_map = extract_team_role_map(info)
        if args.require_perfect_roles and not has_perfect_roles(info):
            filtered_roles += 1
            continue
        if BLUE_TEAM_ID not in role_map or RED_TEAM_ID not in role_map:
            incomplete_role_map += 1
            continue

        frames = frames_upto_minute(get_timeline_frames(timeline), args.max_minute)
        if not frames:
            no_frames += 1
            continue

        total_kept += 1
        match_id = os.path.basename(mdir)

        for team_id in (BLUE_TEAM_ID, RED_TEAM_ID):
            roles = role_map[team_id]
            jungle_pid = roles["JUNGLE"]
            support_pid = roles["UTILITY"]
            adc_pid = roles["BOTTOM"]

            distances: List[float] = []
            river_or_lane_frames = 0
            enemy_jungle_frames = 0
            own_jungle_frames = 0

            for frame_idx, frame in enumerate(frames):
                jungle_pf = get_participant_frame(frame, jungle_pid)
                support_pf = get_participant_frame(frame, support_pid)
                adc_pf = get_participant_frame(frame, adc_pid)

                jungle_pos = extract_position_from_pf(jungle_pf) if participant_is_alive(jungle_pf) else None
                support_pos = extract_position_from_pf(support_pf) if participant_is_alive(support_pf) else None
                adc_pos = extract_position_from_pf(adc_pf) if participant_is_alive(adc_pf) else None

                if jungle_pos is not None:
                    zone = classify_map_zone(jungle_pos[0], jungle_pos[1], team_id)
                    jungle_zone_counter[zone] += 1
                    if zone in LANE_OR_RIVER_ZONES:
                        river_or_lane_frames += 1
                    elif zone in OWN_JUNGLE_ZONES:
                        own_jungle_frames += 1
                    elif zone in {"ENEMY_TOP_JUNGLE", "ENEMY_BOTTOM_JUNGLE"}:
                        enemy_jungle_frames += 1

                    mx, my = mirror_to_blue_side(jungle_pos, team_id, args.mirror_red_to_blue)
                    jungle_xs.append(mx)
                    jungle_ys.append(my)

                    if args.export_jungle_zones_csv:
                        cx, cy = mirror_to_blue_side(jungle_pos, team_id, True)
                        ts_ms = get_frame_timestamp_ms(frame, frame_idx)
                        jungle_zone_rows.append(
                            {
                                "match_id": match_id,
                                "team_id": team_id,
                                "participant_id": jungle_pid,
                                "timestamp_ms": int(round(ts_ms)),
                                "minute": round(ts_ms / 60000.0, 3),
                                "x": round(jungle_pos[0], 1),
                                "y": round(jungle_pos[1], 1),
                                "canonical_x": round(cx, 1),
                                "canonical_y": round(cy, 1),
                                "zone": zone,
                            }
                        )

                if support_pos is not None:
                    mx, my = mirror_to_blue_side(support_pos, team_id, args.mirror_red_to_blue)
                    support_xs.append(mx)
                    support_ys.append(my)

                if support_pos is not None and adc_pos is not None:
                    distances.append(euclidean_distance(support_pos, adc_pos))

            if len(distances) >= args.min_shared_frames:
                support_adc_mean_distances.append(mean(distances))
                duo_team_observations += 1

            active_map_frames = river_or_lane_frames + enemy_jungle_frames
            denom = active_map_frames + own_jungle_frames
            if denom >= args.min_jungle_classified_frames:
                jungle_presence_ratios.append(river_or_lane_frames / denom)
                jungle_team_observations += 1

        if args.max_matches > 0 and total_kept >= args.max_matches:
            print(f"\n--- PRUEBA RÁPIDA: Partidas procesadas: {total_kept} ---")
            break

    print_header("RESUMEN DEL ANÁLISIS ESPACIAL")
    print(f"RAW base: {base}")
    print(f"Partidas vistas: {total_seen}")
    print(f"Bad match.json: {bad_match_json}")
    print(f"Bad timeline.json: {bad_timeline_json}")
    print(f"Sin info en match.json: {missing_info}")
    print(f"Filtradas por duración < {args.min_duration_minutes:.1f} min: {filtered_short}")
    if args.require_perfect_roles:
        print(f"Filtradas por roles no perfectos: {filtered_roles}")
    print(f"Descartadas por no poder mapear roles 1:1: {incomplete_role_map}")
    print(f"Sin frames válidos hasta min {args.max_minute:.1f}: {no_frames}")
    print(f"Partidas analizadas: {total_kept}")
    print(f"Observaciones team-match para Support-ADC: {duo_team_observations}")
    print(f"Observaciones team-match para Jungla: {jungle_team_observations}")
    print(f"Muestras posicionales Jungla: {len(jungle_xs)}")
    print(f"Muestras posicionales Support: {len(support_xs)}")
    if sample_bad_matches:
        print(f"Sample partidas con JSON roto/ilegible: {sample_bad_matches}")

    summarize_distribution(
        "DISTANCIA MEDIA SUPPORT - ADC POR TEAM-MATCH (0-14 min)",
        support_adc_mean_distances,
        unit=" u",
    )
    summarize_distribution(
        "RATIO DE PRESENCIA ACTIVA DEL JUNGLA VS PROPIA JUNGLA",
        jungle_presence_ratios,
    )

    print_header("CONTEO DE ZONAS DEL JUNGLA")
    if jungle_zone_counter:
        for zone, count in jungle_zone_counter.most_common():
            print(f"{zone:<18} {count}")
    else:
        print("Sin observaciones.")

    dist_lines = []
    for q, label in ((0.10, "p10"), (0.25, "p25"), (0.50, "p50"), (0.75, "p75"), (0.90, "p90")):
        value = percentile(support_adc_mean_distances, q)
        if value is not None:
            dist_lines.append((value, f"{label}={value:.0f}"))

    ratio_lines = []
    for q, label in ((0.10, "p10"), (0.25, "p25"), (0.50, "p50"), (0.75, "p75"), (0.90, "p90")):
        value = percentile(jungle_presence_ratios, q)
        if value is not None:
            ratio_lines.append((value, f"{label}={value:.2f}"))

    plot_hexbin_heatmap(
        jungle_xs,
        jungle_ys,
        os.path.join(args.out_dir, "jungle_heatmap_0_14.png"),
        title=f"Heatmap 2D de posiciones de Jungla (0-14 min) ({total_kept} partidas)",
        map_image_path=args.map_image,
    )
    plot_hexbin_heatmap(
        jungle_xs,
        jungle_ys,
        os.path.join(args.out_dir, "jungle_heatmap_0_14_balanced.png"),
        title=f"Heatmap 2D de posiciones de Jungla (0-14 min, balanced) ({total_kept} partidas)",
        map_image_path=args.map_image,
        balanced=True,
        color_cap_percentile=BALANCED_HEATMAP_PERCENTILE,
    )
    plot_hexbin_heatmap(
        support_xs,
        support_ys,
        os.path.join(args.out_dir, "support_heatmap_0_14.png"),
        title=f"Heatmap 2D de posiciones de Support (0-14 min) ({total_kept} partidas)",
        map_image_path=args.map_image,
    )
    plot_hexbin_heatmap(
        support_xs,
        support_ys,
        os.path.join(args.out_dir, "support_heatmap_0_14_balanced.png"),
        title=f"Heatmap 2D de posiciones de Support (0-14 min, balanced) ({total_kept} partidas)",
        map_image_path=args.map_image,
        balanced=True,
        color_cap_percentile=BALANCED_HEATMAP_PERCENTILE,
    )
    plot_histogram(
        support_adc_mean_distances,
        os.path.join(args.out_dir, "support_adc_mean_distance_histogram.png"),
        title="Histograma de distancia media Support-ADC por team-match",
        xlabel="Distancia media 0-14 min (unidades del mapa)",
        bins=45,
        vertical_lines=dist_lines,
    )
    plot_histogram(
        jungle_presence_ratios,
        os.path.join(args.out_dir, "jungle_presence_ratio_histogram.png"),
        title="Histograma del ratio de presencia activa del Jungla",
        xlabel="Ratio = frames(río/líneas) / [frames(río/líneas) + frames(propia jungla)]",
        bins=35,
        xlim=(0.0, 1.0),
        vertical_lines=ratio_lines,
    )
    plot_counter(
        jungle_zone_counter,
        os.path.join(args.out_dir, "jungle_zone_counts.png"),
        title="Conteo bruto de zonas del Jungla (0-14 min)",
        xlabel="Zona",
        ylabel="Número de frames",
    )
    plot_zone_reference(
        os.path.join(args.out_dir, "zone_reference_map.png"),
        map_image_path=args.map_image,
    )
    plot_zone_reference(
        os.path.join(args.out_dir, "zone_reference_on_jungle_heatmap.png"),
        map_image_path=args.map_image,
        background_xs=jungle_xs,
        background_ys=jungle_ys,
        balanced_background=True,
        heatmap_alpha=0.62,
        zone_alpha=0.24,
    )

    if args.export_jungle_zones_csv:
        csv_path = os.path.join(args.out_dir, "jungle_frame_zones.csv")
        write_zone_rows_csv(jungle_zone_rows, csv_path)
        print(csv_path)

    print_header("PNG GENERADOS")
    print(os.path.join(args.out_dir, "jungle_heatmap_0_14.png"))
    print(os.path.join(args.out_dir, "jungle_heatmap_0_14_balanced.png"))
    print(os.path.join(args.out_dir, "support_heatmap_0_14.png"))
    print(os.path.join(args.out_dir, "support_heatmap_0_14_balanced.png"))
    print(os.path.join(args.out_dir, "support_adc_mean_distance_histogram.png"))
    print(os.path.join(args.out_dir, "jungle_presence_ratio_histogram.png"))
    print(os.path.join(args.out_dir, "jungle_zone_counts.png"))
    print(os.path.join(args.out_dir, "zone_reference_map.png"))
    print(os.path.join(args.out_dir, "zone_reference_on_jungle_heatmap.png"))


if __name__ == "__main__":
    main()
