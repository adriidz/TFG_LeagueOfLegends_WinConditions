import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_RAW_ROOT = os.path.join("Data_raw", "raw")
DEFAULT_REGION = "europe"
DEFAULT_OUT_PATH = os.path.join("Data_clean", "labels", "jungle_labels.parquet")
CANONICAL_ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
BLUE_TEAM_ID = 100
RED_TEAM_ID = 200
MAP_MAX = 15000.0
DEFAULT_MAX_MINUTE = 10.0
EMPTY_ROLE_VALUES = {None, "", "INVALID", "NONE"}

# Geometría aproximada heredada del análisis exploratorio.
MID_HALF_WIDTH = 1200.0
RIVER_HALF_WIDTH = 1300.0
EDGE_LANE_BAND = 3200.0
BASE_CORNER = 3500.0
INNER_MARGIN = 1400.0
DRAGON_PIT_CENTER = (10500.0, 4500.0)
BARON_PIT_CENTER = (4500.0, 10500.0)
PIT_RADIUS = 1700.0

LANE_OR_RIVER_ZONES = {"TOP", "MIDDLE", "BOTTOM", "RIVER"}
JUNGLE_LABEL_COLUMN_ORDER = [
    "match_id",
    "team_id",
    "side",
    "patch",
    "game_version",
    "game_start_timestamp",
    "max_minute",
    "jungle_participant_id",
    "jungle_champion_id",
    "jungle_champion_name",
    "valid_jungle_frames",
    "river_frames",
    "lane_frames",
    "river_or_lane_frames",
    "own_jungle_frames",
    "enemy_jungle_frames",
    "base_frames",
    "other_frames",
    "jungle_presence_score",
    "jungle_presence_label",
]


@dataclass
class TeamJungleMetrics:
    match_id: str
    team_id: int
    side: str
    patch: Optional[str]
    game_version: Optional[str]
    game_start_timestamp: Optional[int]
    max_minute: float
    jungle_participant_id: int
    jungle_champion_id: Optional[int]
    jungle_champion_name: Optional[str]
    valid_jungle_frames: int
    river_frames: int
    lane_frames: int
    river_or_lane_frames: int
    own_jungle_frames: int
    enemy_jungle_frames: int
    base_frames: int
    other_frames: int
    jungle_presence_score: Optional[float]
    jungle_presence_label: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "team_id": self.team_id,
            "side": self.side,
            "patch": self.patch,
            "game_version": self.game_version,
            "game_start_timestamp": self.game_start_timestamp,
            "max_minute": self.max_minute,
            "jungle_participant_id": self.jungle_participant_id,
            "jungle_champion_id": self.jungle_champion_id,
            "jungle_champion_name": self.jungle_champion_name,
            "valid_jungle_frames": self.valid_jungle_frames,
            "river_frames": self.river_frames,
            "lane_frames": self.lane_frames,
            "river_or_lane_frames": self.river_or_lane_frames,
            "own_jungle_frames": self.own_jungle_frames,
            "enemy_jungle_frames": self.enemy_jungle_frames,
            "base_frames": self.base_frames,
            "other_frames": self.other_frames,
            "jungle_presence_score": self.jungle_presence_score,
            "jungle_presence_label": self.jungle_presence_label,
        }


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def list_match_dirs(base: str) -> List[str]:
    if not os.path.isdir(base):
        raise SystemExit(f"No existe el directorio RAW: {base}")
    out: List[str] = []
    for name in os.listdir(base):
        mdir = os.path.join(base, name)
        if os.path.isdir(mdir):
            out.append(mdir)
    out.sort()
    return out


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_match_info(match: dict) -> dict:
    return (match or {}).get("info") or {}


def get_match_id(match: dict, match_dir: str) -> str:
    metadata = (match or {}).get("metadata") or {}
    match_id = metadata.get("matchId")
    if match_id:
        return str(match_id)
    info = get_match_info(match)
    game_id = info.get("gameId")
    if game_id is not None:
        return str(game_id)
    return os.path.basename(match_dir)


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

    if x <= 1.0 and y <= 1.0:
        return False

    return True


def participant_lookup(info: dict) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for participant in info.get("participants") or []:
        pid = participant.get("participantId")
        if isinstance(pid, int):
            out[pid] = participant
    return out


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


def is_mid_lane(x: float, y: float) -> bool:
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
    if is_top_lane(x, y) and y >= x:
        return "TOP"
    if is_bottom_lane(x, y) and x >= y:
        return "BOTTOM"

    side_value = x + y
    if team_id == BLUE_TEAM_ID:
        return "OWN_JUNGLE" if side_value < MAP_MAX else "ENEMY_JUNGLE"
    return "OWN_JUNGLE" if side_value > MAP_MAX else "ENEMY_JUNGLE"


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


def compute_jungle_presence_score(river_or_lane_frames: int, own_jungle_frames: int) -> Optional[float]:
    denom = river_or_lane_frames + own_jungle_frames
    if denom <= 0:
        return None
    return river_or_lane_frames / denom


def build_team_jungle_metrics(
    *,
    match_id: str,
    team_id: int,
    info: dict,
    frames: List[dict],
    role_map: Dict[int, Dict[str, int]],
    min_jungle_classified_frames: int,
    max_minute: float,
) -> Optional[TeamJungleMetrics]:
    participants = participant_lookup(info)
    jungle_pid = role_map[team_id]["JUNGLE"]
    jungle_pf_meta = participants.get(jungle_pid, {})

    river_frames = 0
    lane_frames = 0
    own_jungle_frames = 0
    enemy_jungle_frames = 0
    base_frames = 0
    other_frames = 0
    valid_jungle_frames = 0

    for frame in frames:
        jungle_pf = get_participant_frame(frame, jungle_pid)
        if not participant_is_alive(jungle_pf):
            continue

        jungle_pos = extract_position_from_pf(jungle_pf)
        if jungle_pos is None:
            continue

        valid_jungle_frames += 1
        zone = classify_map_zone(jungle_pos[0], jungle_pos[1], team_id)

        if zone == "RIVER":
            river_frames += 1
        elif zone in {"TOP", "MIDDLE", "BOTTOM"}:
            lane_frames += 1
        elif zone == "OWN_JUNGLE":
            own_jungle_frames += 1
        elif zone == "ENEMY_JUNGLE":
            enemy_jungle_frames += 1
        elif zone == "BASE":
            base_frames += 1
        else:
            other_frames += 1

    river_or_lane_frames = river_frames + lane_frames
    classified_for_score = river_or_lane_frames + own_jungle_frames
    if classified_for_score < min_jungle_classified_frames:
        return None

    score = compute_jungle_presence_score(river_or_lane_frames, own_jungle_frames)
    if score is None:
        return None

    return TeamJungleMetrics(
        match_id=match_id,
        team_id=team_id,
        side="blue" if team_id == BLUE_TEAM_ID else "red",
        patch=info.get("gameVersion"),
        game_version=info.get("gameVersion"),
        game_start_timestamp=info.get("gameStartTimestamp") or info.get("gameCreation"),
        max_minute=max_minute,
        jungle_participant_id=jungle_pid,
        jungle_champion_id=jungle_pf_meta.get("championId"),
        jungle_champion_name=jungle_pf_meta.get("championName"),
        valid_jungle_frames=valid_jungle_frames,
        river_frames=river_frames,
        lane_frames=lane_frames,
        river_or_lane_frames=river_or_lane_frames,
        own_jungle_frames=own_jungle_frames,
        enemy_jungle_frames=enemy_jungle_frames,
        base_frames=base_frames,
        other_frames=other_frames,
        jungle_presence_score=score,
    )


def apply_quantile_labels(df: pd.DataFrame, lower_q: float, upper_q: float) -> Tuple[pd.DataFrame, Optional[float], Optional[float]]:
    if df.empty:
        df["jungle_presence_label"] = pd.Series(dtype="object")
        return df, None, None

    valid_scores = df["jungle_presence_score"].dropna()
    if valid_scores.empty:
        df["jungle_presence_label"] = None
        return df, None, None

    lower_thr = float(valid_scores.quantile(lower_q))
    upper_thr = float(valid_scores.quantile(upper_q))

    labels: List[Optional[str]] = []
    for score in df["jungle_presence_score"]:
        if pd.isna(score):
            labels.append(None)
        elif float(score) <= lower_thr:
            labels.append("farm_oriented")
        elif float(score) >= upper_thr:
            labels.append("map_presence")
        else:
            labels.append("ambiguous")

    df = df.copy()
    df["jungle_presence_label"] = labels
    return df, lower_thr, upper_thr


def build_percentile_table(scores: pd.Series) -> pd.DataFrame:
    percentiles = [0.01, 0.05, 0.10, 0.25, 0.33, 0.50, 0.66, 0.75, 0.90, 0.95, 0.99]
    rows = []
    for q in percentiles:
        rows.append({
            "percentile": q,
            "score": float(scores.quantile(q)),
        })
    return pd.DataFrame(rows)


def summarize_by_champion(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df["jungle_presence_score"].notna()].copy()
    if work.empty:
        return pd.DataFrame()

    grouped = work.groupby(["jungle_champion_id", "jungle_champion_name"], dropna=False)
    summary = grouped["jungle_presence_score"].agg(
        n="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
        q10=lambda s: s.quantile(0.10),
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
        q90=lambda s: s.quantile(0.90),
    ).reset_index()
    summary = summary.sort_values(["n", "mean"], ascending=[False, False]).reset_index(drop=True)
    return summary


def summarize_by_side(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df["jungle_presence_score"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    summary = work.groupby("side")["jungle_presence_score"].agg(
        n="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
    ).reset_index()
    return summary.sort_values("side").reset_index(drop=True)


def save_dataframe(df: pd.DataFrame, path_no_ext: str) -> None:
    ensure_parent_dir(path_no_ext + ".parquet")
    df.to_parquet(path_no_ext + ".parquet", index=False)
    df.to_csv(path_no_ext + ".csv", index=False)


def maybe_plot_score_distribution(scores: pd.Series, out_png: str, title: str) -> None:
    if scores.empty:
        return
    ensure_parent_dir(out_png)
    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=40)
    plt.xlabel("jungle_presence_score")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def maybe_plot_top_champions(summary: pd.DataFrame, out_png: str, min_games: int, top_n: int) -> None:
    if summary.empty:
        return
    filtered = summary[summary["n"] >= min_games].copy()
    if filtered.empty:
        return
    filtered = filtered.sort_values(["mean", "n"], ascending=[False, False]).head(top_n)
    if filtered.empty:
        return

    labels = filtered["jungle_champion_name"].fillna("UNKNOWN").astype(str)
    values = filtered["mean"].astype(float)

    ensure_parent_dir(out_png)
    plt.figure(figsize=(10, max(4, 0.4 * len(filtered))))
    plt.barh(labels.iloc[::-1], values.iloc[::-1])
    plt.xlabel("mean jungle_presence_score")
    plt.ylabel("champion")
    plt.title(f"Top {len(filtered)} champions by map presence (min {min_games} games)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def maybe_plot_bottom_champions(summary: pd.DataFrame, out_png: str, min_games: int, top_n: int) -> None:
    if summary.empty:
        return
    filtered = summary[summary["n"] >= min_games].copy()
    if filtered.empty:
        return
    filtered = filtered.sort_values(["mean", "n"], ascending=[True, False]).head(top_n)
    if filtered.empty:
        return

    labels = filtered["jungle_champion_name"].fillna("UNKNOWN").astype(str)
    values = filtered["mean"].astype(float)

    ensure_parent_dir(out_png)
    plt.figure(figsize=(10, max(4, 0.4 * len(filtered))))
    plt.barh(labels.iloc[::-1], values.iloc[::-1])
    plt.xlabel("mean jungle_presence_score")
    plt.ylabel("champion")
    plt.title(f"Bottom {len(filtered)} champions by map presence (min {min_games} games)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def print_overall_summary(df: pd.DataFrame, lower_thr: Optional[float], upper_thr: Optional[float]) -> None:
    valid_scores = df["jungle_presence_score"].dropna()
    if valid_scores.empty:
        print("\nNo hay scores válidos para resumir.")
        return

    print("\nDistribución jungle_presence_score")
    print(valid_scores.describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).to_string())
    if lower_thr is not None and upper_thr is not None:
        print(f"\nThresholds de discretización: lower={lower_thr:.6f} | upper={upper_thr:.6f}")

    label_counts = df["jungle_presence_label"].value_counts(dropna=False)
    if not label_counts.empty:
        print("\nConteo de jungle_presence_label")
        print(label_counts.to_string())


def print_champion_preview(summary: pd.DataFrame, min_games: int, top_n: int) -> None:
    if summary.empty:
        return
    filtered = summary[summary["n"] >= min_games].copy()
    if filtered.empty:
        print(f"\nNo hay campeones con al menos {min_games} observaciones para preview.")
        return

    print(f"\nTop {top_n} campeones por jungle_presence_score medio (n >= {min_games})")
    print(filtered.sort_values(["mean", "n"], ascending=[False, False]).head(top_n).to_string(index=False))

    print(f"\nBottom {top_n} campeones por jungle_presence_score medio (n >= {min_games})")
    print(filtered.sort_values(["mean", "n"], ascending=[True, False]).head(top_n).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye etiquetas continuas del jungla a nivel (match_id, team_id), las guarda en Parquet y genera resúmenes automáticos."
    )
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--out-path", default=DEFAULT_OUT_PATH)
    parser.add_argument("--min-duration-minutes", type=float, default=15.0)
    parser.add_argument("--max-minute", type=float, default=DEFAULT_MAX_MINUTE)
    parser.add_argument("--min-jungle-classified-frames", type=int, default=8)
    parser.add_argument(
        "--lower-quantile",
        type=float,
        default=0.33,
        help="Cuantil inferior para discretizar la etiqueta (farm_oriented).",
    )
    parser.add_argument(
        "--upper-quantile",
        type=float,
        default=0.66,
        help="Cuantil superior para discretizar la etiqueta (map_presence).",
    )
    parser.add_argument(
        "--drop-ambiguous",
        action="store_true",
        help="Si se activa, elimina las observaciones ambiguas antes de guardar.",
    )
    parser.add_argument(
        "--require-perfect-roles",
        action="store_true",
        default=True,
        help="Exige un mapeo 1:1 perfecto de roles canónicos en ambos equipos.",
    )
    parser.add_argument(
        "--allow-imperfect-roles",
        dest="require_perfect_roles",
        action="store_false",
        help="Permite seguir aunque el mapeo no sea perfecto; se descartará igualmente cada equipo no mapeable.",
    )
    parser.add_argument(
        "--analysis-dir",
        default=None,
        help="Directorio donde guardar automáticamente tablas de percentiles, resúmenes por campeón y gráficas. Por defecto usa el mismo directorio del out-path.",
    )
    parser.add_argument(
        "--champion-min-games",
        type=int,
        default=300,
        help="Mínimo de observaciones por campeón para rankings resumidos y gráficas de campeones.",
    )
    parser.add_argument(
        "--preview-top-n",
        type=int,
        default=15,
        help="Número de campeones a mostrar en el preview de top/bottom.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Si se activa, no genera PNGs de distribución ni rankings.",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=0,
        help=(
            "Máximo número de partidas a procesar "
            "(default: 10000, usa 0 para procesar todas)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.lower_quantile < args.upper_quantile < 1.0):
        raise SystemExit("Los cuantiles deben cumplir 0 < lower < upper < 1.")
    if args.champion_min_games < 1:
        raise SystemExit("--champion-min-games debe ser >= 1.")
    if args.preview_top_n < 1:
        raise SystemExit("--preview-top-n debe ser >= 1.")

    raw_base = os.path.join(args.raw_root, args.region)
    match_dirs = list_match_dirs(raw_base)
    print(f"Directorios de partida detectados: {len(match_dirs)}")
    if args.max_matches and args.max_matches > 0:
        match_dirs = match_dirs[: args.max_matches]
        print(f"Directorios de partida a procesar: {len(match_dirs)}")
    ensure_parent_dir(args.out_path)

    out_stem = os.path.splitext(args.out_path)[0]
    analysis_dir = args.analysis_dir or os.path.join(os.path.dirname(args.out_path), os.path.basename(out_stem) + "_analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    total_seen = 0
    total_kept = 0
    bad_match_json = 0
    bad_timeline_json = 0
    missing_info = 0
    filtered_short = 0
    filtered_roles = 0
    no_frames = 0
    missing_team_role_map = 0
    team_metrics_rejected = 0

    rows: List[dict] = []
    progress_every = 1000

    t0 = time.time()
    last_log = t0

    for mdir in match_dirs:
        total_seen += 1

        if total_seen % progress_every == 0 or (time.time() - last_log) > 15:
            now = time.time()
            elapsed = now - t0
            rate = total_seen / elapsed if elapsed > 0 else 0.0
            print(
                f"[{total_seen}/{len(match_dirs)}] "
                f"kept_matches={total_kept} "
                f"rows={len(rows)} "
                f"bad_match={bad_match_json} "
                f"bad_tl={bad_timeline_json} "
                f"short={filtered_short} "
                f"roles={filtered_roles + missing_team_role_map} "
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
            continue

        try:
            timeline = load_json(timeline_path)
        except Exception:
            bad_timeline_json += 1
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
        if args.require_perfect_roles and not (BLUE_TEAM_ID in role_map and RED_TEAM_ID in role_map):
            filtered_roles += 1
            continue

        frames = frames_upto_minute(get_timeline_frames(timeline), args.max_minute)
        if not frames:
            no_frames += 1
            continue

        total_kept += 1
        match_id = get_match_id(match, mdir)

        for team_id in (BLUE_TEAM_ID, RED_TEAM_ID):
            if team_id not in role_map:
                missing_team_role_map += 1
                continue

            metrics = build_team_jungle_metrics(
                match_id=match_id,
                team_id=team_id,
                info=info,
                frames=frames,
                role_map=role_map,
                min_jungle_classified_frames=args.min_jungle_classified_frames,
                max_minute=args.max_minute,
            )
            if metrics is None:
                team_metrics_rejected += 1
                continue
            rows.append(metrics.to_dict())

    df = pd.DataFrame(rows)
    if df.empty:
        print("No se generaron observaciones válidas. Revisa filtros y rutas.")
        return

    df, lower_thr, upper_thr = apply_quantile_labels(df, args.lower_quantile, args.upper_quantile)
    if args.drop_ambiguous:
        df = df[df["jungle_presence_label"] != "ambiguous"].copy()

    for col in JUNGLE_LABEL_COLUMN_ORDER:
        if col not in df.columns:
            df[col] = None
    df = df[JUNGLE_LABEL_COLUMN_ORDER].sort_values(["match_id", "team_id"]).reset_index(drop=True)

    df.to_parquet(args.out_path, index=False)

    valid_scores = df["jungle_presence_score"].dropna()
    percentile_table = build_percentile_table(valid_scores) if not valid_scores.empty else pd.DataFrame()
    champion_summary = summarize_by_champion(df)
    side_summary = summarize_by_side(df)

    save_dataframe(percentile_table, os.path.join(analysis_dir, "overall_percentiles"))
    save_dataframe(champion_summary, os.path.join(analysis_dir, "champion_summary"))
    save_dataframe(side_summary, os.path.join(analysis_dir, "side_summary"))

    if not args.skip_plots and not valid_scores.empty:
        maybe_plot_score_distribution(
            valid_scores,
            os.path.join(analysis_dir, "score_distribution.png"),
            f"Jungle presence score distribution (0-{args.max_minute:g} min)",
        )
        maybe_plot_top_champions(
            champion_summary,
            os.path.join(analysis_dir, "top_champions_map_presence.png"),
            min_games=args.champion_min_games,
            top_n=args.preview_top_n,
        )
        maybe_plot_bottom_champions(
            champion_summary,
            os.path.join(analysis_dir, "bottom_champions_map_presence.png"),
            min_games=args.champion_min_games,
            top_n=args.preview_top_n,
        )

    print("\n============================")
    print("RESUMEN BUILD_JUNGLE_LABELS")
    print("============================")
    print(f"RAW base: {raw_base}")
    print(f"Partidas vistas: {total_seen}")
    print(f"Partidas analizadas: {total_kept}")
    print(f"Bad match.json: {bad_match_json}")
    print(f"Bad timeline.json: {bad_timeline_json}")
    print(f"Sin info en match.json: {missing_info}")
    print(f"Filtradas por duración < {args.min_duration_minutes:.1f} min: {filtered_short}")
    print(f"Filtradas por roles: {filtered_roles}")
    print(f"Equipos sin mapeo utilizable: {missing_team_role_map}")
    print(f"Partidas sin frames válidos hasta min {args.max_minute:.1f}: {no_frames}")
    print(f"Observaciones team-match descartadas por score no computable: {team_metrics_rejected}")
    print(f"Filas guardadas: {len(df)}")
    print(f"Output parquet: {args.out_path}")
    print(f"Analysis dir: {analysis_dir}")

    print_overall_summary(df, lower_thr=lower_thr, upper_thr=upper_thr)
    if not side_summary.empty:
        print("\nResumen por side")
        print(side_summary.to_string(index=False))
    print_champion_preview(champion_summary, min_games=args.champion_min_games, top_n=args.preview_top_n)


if __name__ == "__main__":
    main()
