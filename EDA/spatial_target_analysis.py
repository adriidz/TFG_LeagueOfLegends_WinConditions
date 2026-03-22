import argparse
import json
import math
import os
import time
from collections import Counter
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

DEFAULT_RAW_ROOT = os.path.join("Data_raw", "raw")
DEFAULT_REGION = "europe"
DEFAULT_OUT_DIR = os.path.join("Data_clean", "spatial_reports")
CANONICAL_ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
BLUE_TEAM_ID = 100
RED_TEAM_ID = 200
MAP_MIN = 0.0
MAP_MAX = 15000.0
DEFAULT_MAX_MINUTE = 14.0
EMPTY_ROLE_VALUES = {None, "", "INVALID", "NONE"}

# Geometría aproximada de la Grieta para un primer análisis exploratorio.
MID_HALF_WIDTH = 1200.0
RIVER_HALF_WIDTH = 1300.0
EDGE_LANE_BAND = 3200.0
# Más grande para cubrir fuente + rampa y evitar clasificar base como lane/mid.
BASE_CORNER = 3500.0
# Más conservador para no contar esquinas/bases como si fueran zonas jugables centrales.
INNER_MARGIN = 1400.0
# Fosos aproximados para "engordar" el río alrededor de objetivos neutrales.
DRAGON_PIT_CENTER = (10500.0, 4500.0)
BARON_PIT_CENTER = (4500.0, 10500.0)
PIT_RADIUS = 1700.0

# Ajustes visuales para heatmaps "balanced"
BALANCED_HEATMAP_PERCENTILE = 0.99    # 99%
BALANCED_BORDER_MARGIN = 700.0        # recorta borde extremo
BALANCED_CORNER_MASK = 1600.0         # recorta base profunda / esquina enemiga

LANE_OR_RIVER_ZONES = {"TOP", "MIDDLE", "BOTTOM", "RIVER"}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
            if value > 100000:  # defensivo: milisegundos
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


def mirror_to_blue_side(pos: Tuple[float, float], team_id: int, mirror_red: bool) -> Tuple[float, float]:
    x, y = pos
    if mirror_red and team_id == RED_TEAM_ID:
        return MAP_MAX - x, MAP_MAX - y
    return x, y


def is_in_base(x: float, y: float, team_id: int) -> bool:
    if team_id == BLUE_TEAM_ID:
        return x <= BASE_CORNER and y <= BASE_CORNER
    return x >= MAP_MAX - BASE_CORNER and y >= MAP_MAX - BASE_CORNER


def is_top_lane(x: float, y: float) -> bool:
    return x <= EDGE_LANE_BAND or y >= MAP_MAX - EDGE_LANE_BAND


def is_bottom_lane(x: float, y: float) -> bool:
    return y <= EDGE_LANE_BAND or x >= MAP_MAX - EDGE_LANE_BAND


def is_near_point(x: float, y: float, center: Tuple[float, float], radius: float) -> bool:
    cx, cy = center
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2


def participant_is_alive(participant_frame: Optional[dict]) -> bool:
    """
    Filtro defensivo:
    - si el timeline trae un flag explícito de vida, lo usamos;
    - si trae HP actual, lo usamos;
    - fallback: descartamos posiciones nulas o (0,0), que suelen ser basura útilmente filtrable.
    """
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

    if x <= 1.0 and y <= 1.0:
        return False

    return True


def is_mid_lane(x: float, y: float) -> bool:
    # Evita que base / rampas profundas entren como "mid" solo por estar cerca de x=y.
    in_playable_center = (
        BASE_CORNER <= x <= MAP_MAX - BASE_CORNER
        and BASE_CORNER <= y <= MAP_MAX - BASE_CORNER
    )
    return in_playable_center and abs(x - y) <= MID_HALF_WIDTH


def is_river(x: float, y: float) -> bool:
    in_bounds = (
        INNER_MARGIN <= x <= MAP_MAX - INNER_MARGIN
        and INNER_MARGIN <= y <= MAP_MAX - INNER_MARGIN
    )
    if not in_bounds:
        return False

    on_main_band = abs((x + y) - MAP_MAX) <= RIVER_HALF_WIDTH
    in_dragon_pit = is_near_point(x, y, DRAGON_PIT_CENTER, PIT_RADIUS)
    in_baron_pit = is_near_point(x, y, BARON_PIT_CENTER, PIT_RADIUS)

    return on_main_band or in_dragon_pit or in_baron_pit


def classify_map_zone(x: float, y: float, team_id: int) -> str:
    if is_in_base(x, y, team_id):
        return "BASE"
    if is_river(x, y):
        return "RIVER"
    if is_mid_lane(x, y):
        return "MIDDLE"
    # Las side lanes van después de río y mid para evitar que invadan demasiado.
    if is_top_lane(x, y) and y >= x:
        return "TOP"
    if is_bottom_lane(x, y) and x >= y:
        return "BOTTOM"

    # Partimos la jungla por lado del río (x + y = 15000 aprox.)
    side_value = x + y
    if team_id == BLUE_TEAM_ID:
        return "OWN_JUNGLE" if side_value < MAP_MAX else "ENEMY_JUNGLE"
    return "OWN_JUNGLE" if side_value > MAP_MAX else "ENEMY_JUNGLE"


def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


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
    # Fuera del área jugable razonable
    if x < border_margin or y < border_margin:
        return False
    if x > MAP_MAX - border_margin or y > MAP_MAX - border_margin:
        return False

    # Base/fuente propia profunda (tras mirror, suele quedar en bottom-left)
    if x <= corner_mask and y <= corner_mask:
        return False

    # Base enemiga profunda
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
        alpha = 0.55
    else:
        alpha = 0.9

    hb = plt.hexbin(
        plot_xs,
        plot_ys,
        gridsize=55,
        extent=(MAP_MIN, MAP_MAX, MAP_MIN, MAP_MAX),
        mincnt=1,
        alpha=alpha,
        norm=mcolors.LogNorm(vmin=1.0),
    )

    if color_cap_percentile is not None:
        counts = [float(v) for v in hb.get_array() if float(v) > 0.0]
        vmax = percentile(counts, color_cap_percentile)
        if vmax is not None and vmax > 1.0:
            hb.set_clim(1.0, vmax)

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

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
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
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    base = os.path.join(args.raw_root, args.region)
    match_dirs = list_match_dirs(base)

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

        for team_id in (BLUE_TEAM_ID, RED_TEAM_ID):
            roles = role_map[team_id]
            jungle_pid = roles["JUNGLE"]
            support_pid = roles["UTILITY"]
            adc_pid = roles["BOTTOM"]

            distances: List[float] = []
            river_or_lane_frames = 0
            own_jungle_frames = 0

            for frame in frames:
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
                    elif zone == "OWN_JUNGLE":
                        own_jungle_frames += 1

                    mx, my = mirror_to_blue_side(jungle_pos, team_id, args.mirror_red_to_blue)
                    jungle_xs.append(mx)
                    jungle_ys.append(my)

                if support_pos is not None:
                    mx, my = mirror_to_blue_side(support_pos, team_id, args.mirror_red_to_blue)
                    support_xs.append(mx)
                    support_ys.append(my)

                if support_pos is not None and adc_pos is not None:
                    distances.append(euclidean_distance(support_pos, adc_pos))

            if len(distances) >= args.min_shared_frames:
                support_adc_mean_distances.append(mean(distances))
                duo_team_observations += 1

            denom = river_or_lane_frames + own_jungle_frames
            if denom >= args.min_jungle_classified_frames:
                jungle_presence_ratios.append(river_or_lane_frames / denom)
                jungle_team_observations += 1

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
        "RATIO DE PRESENCIA DEL JUNGLA EN RÍO/LÍNEAS VS PROPIA JUNGLA",
        jungle_presence_ratios,
    )

    print_header("CONTEO DE ZONAS DEL JUNGLA")
    if jungle_zone_counter:
        for zone, count in jungle_zone_counter.most_common():
            print(f"{zone:<13} {count}")
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
        title="Heatmap 2D de posiciones de Jungla (0-14 min)",
        map_image_path=args.map_image,
    )
        
    plot_hexbin_heatmap(
        jungle_xs,
        jungle_ys,
        os.path.join(args.out_dir, "jungle_heatmap_0_14_balanced.png"),
        title="Heatmap 2D de posiciones de Jungla (0-14 min, balanced)",
        map_image_path=args.map_image,
        balanced=True,
        color_cap_percentile=BALANCED_HEATMAP_PERCENTILE,
    )

    plot_hexbin_heatmap(
        support_xs,
        support_ys,
        os.path.join(args.out_dir, "support_heatmap_0_14.png"),
        title="Heatmap 2D de posiciones de Support (0-14 min)",
        map_image_path=args.map_image,
    )
    plot_hexbin_heatmap(
        support_xs,
        support_ys,
        os.path.join(args.out_dir, "support_heatmap_0_14_balanced.png"),
        title="Heatmap 2D de posiciones de Support (0-14 min, balanced)",
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
        title="Histograma del ratio de presencia del Jungla en río/líneas",
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

    print_header("PNG GENERADOS")
    print(os.path.join(args.out_dir, "jungle_heatmap_0_14.png"))
    print(os.path.join(args.out_dir, "jungle_heatmap_0_14_balanced.png"))
    print(os.path.join(args.out_dir, "support_heatmap_0_14.png"))
    print(os.path.join(args.out_dir, "support_heatmap_0_14_balanced.png"))
    print(os.path.join(args.out_dir, "support_adc_mean_distance_histogram.png"))
    print(os.path.join(args.out_dir, "jungle_presence_ratio_histogram.png"))
    print(os.path.join(args.out_dir, "jungle_zone_counts.png"))


if __name__ == "__main__":
    main()
