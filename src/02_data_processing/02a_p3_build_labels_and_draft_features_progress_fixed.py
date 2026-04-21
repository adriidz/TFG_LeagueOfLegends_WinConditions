#!/usr/bin/env python3
"""
02a_build_labels_and_draft_features.py

Single-pass builder: lee cada match.json + timeline.json UNA sola vez
y produce en paralelo:
  - jungle_labels.parquet
  - support_labels.parquet
  - team_tendency_labels.parquet
  - draft_features.parquet   (incluye runas y hechizos de invocador)

Reemplaza los scripts individuales:
  02a_build_jungle_labels.py,  02a_build_support_labels.py,
  02a_build_team_tendency_labels.py,  02b_build_draft_features.py.

Labeling-mode por defecto: by_side (mitiga asimetría blue/red).

Fase 1: permite generar varias ventanas temporales de labels en UNA sola pasada
sobre raw/timeline, para comparar predictibilidad 0-6, 0-8, 0-10, etc.
Draft features se genera una sola vez, ya que no depende de la ventana.

Fase 2: permite elegir el esquema de discretización de labels:
- ternary: extremo / ambiguous / extremo
- binary_clean: conserva solo extremos y descarta la banda central
- binary_full: fuerza todo el espacio a dos clases usando un split intermedio
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from shared_utils import (
    BLUE_TEAM_ID, RED_TEAM_ID, CANONICAL_ROLES, ROLE_KEYS_LOWER,
    OWN_JUNGLE_ZONES, ENEMY_JUNGLE_ZONES, BASE_ZONES, BOT_SIDE_ZONES, RIVER_ZONES,
    DEFAULT_MIN_DURATION_MINUTES, DEFAULT_MAX_MINUTE,
    classify_map_zone, classify_team_side,
    load_json, get_match_info, get_match_id, get_timeline_frames,
    game_duration_minutes, safe_game_duration_seconds, infer_patch,
    extract_team_role_map, participant_lookup,
    get_participant_frame, extract_position, participant_is_alive,
    extract_summoner_spells, extract_runes, extract_team_bans,
    frames_in_window, side_from_team_id,
    ensure_parent_dir, ensure_dir, apply_sample_suffix, get_target_frac,
    list_match_dirs, validate_no_duplicate_keys, save_dataframe,
)

# ── RUTAS ────────────────────────────────────────────────────────────────────
DEFAULT_RAW_ROOT = os.path.join("data", "raw", "raw")
DEFAULT_REGION = "europe"
DEFAULT_OUT_DIR_LABELS = os.path.join("data", "clean", "labels")
DEFAULT_OUT_DIR_FEATURES = os.path.join("data", "clean", "features")

JUNGLE_OUT = "jungle_labels"
SUPPORT_OUT = "support_labels"
TEAM_OUT = "team_tendency_labels"
DRAFT_OUT = "draft_features"


def format_window_tag(max_minute: float) -> str:
    rounded = int(round(float(max_minute)))
    return f"m{rounded:02d}"


def apply_window_suffix(path: str, max_minute: Optional[float], use_suffix: bool) -> str:
    if not use_suffix or max_minute is None:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_{format_window_tag(max_minute)}{ext}"



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
    path: str,
    lower_q: Optional[float],
    upper_q: Optional[float],
    lower_thr: Optional[float],
    upper_thr: Optional[float],
) -> str:
    base, ext = os.path.splitext(path)
    tag = format_quantile_or_threshold_tag(lower_q, upper_q, lower_thr, upper_thr)
    return f"{base}_{tag}{ext}"


def parse_analysis_windows(args: argparse.Namespace) -> List[float]:
    if getattr(args, "analysis_max_minutes", None):
        windows = sorted({float(x) for x in args.analysis_max_minutes if float(x) > 0})
        if not windows:
            raise SystemExit("--analysis-max-minutes debe contener al menos una ventana positiva.")
        return windows
    return [float(args.max_minute)]


def build_window_output_paths(out_labels_dir: str, base_suffix: str, max_minute: float, multi_window: bool) -> Dict[str, str]:
    jg_path = apply_window_suffix(os.path.join(out_labels_dir, f"{JUNGLE_OUT}{base_suffix}.parquet"), max_minute, multi_window)
    sp_path = apply_window_suffix(os.path.join(out_labels_dir, f"{SUPPORT_OUT}{base_suffix}.parquet"), max_minute, multi_window)
    tm_path = apply_window_suffix(os.path.join(out_labels_dir, f"{TEAM_OUT}{base_suffix}.parquet"), max_minute, multi_window)
    return {
        "jungle": jg_path,
        "support": sp_path,
        "team": tm_path,
        "jungle_analysis": os.path.splitext(jg_path)[0] + "_analysis",
        "support_analysis": os.path.splitext(sp_path)[0] + "_analysis",
        "team_analysis": os.path.splitext(tm_path)[0] + "_analysis",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ARGPARSE
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-pass builder: 3 labels + draft features."
    )
    # Filtros generales
    p.add_argument("--min-duration-minutes", type=float, default=DEFAULT_MIN_DURATION_MINUTES)
    p.add_argument("--max-minute", type=float, default=DEFAULT_MAX_MINUTE)
    p.add_argument(
        "--analysis-max-minutes",
        nargs="+",
        type=float,
        default=None,
        help="Ventanas temporales para Fase 1. Si se pasa, genera labels para cada ventana en una sola pasada (ej: 6 8 10 12 15).",
    )
    p.add_argument("--max-matches", type=int, default=0)
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--raw-root", default=DEFAULT_RAW_ROOT,
                   help="Directorio base de raw matches. Debe contener subcarpetas por región.")
    p.add_argument("--region", default=DEFAULT_REGION,
                   help="Región dentro de --raw-root (ej. europe).")
    p.add_argument("--out-labels-dir", default=DEFAULT_OUT_DIR_LABELS,
                   help="Directorio de salida para labels parquet y análisis asociados.")
    p.add_argument("--out-features-dir", default=DEFAULT_OUT_DIR_FEATURES,
                   help="Directorio de salida para draft_features parquet y análisis asociados.")
    p.add_argument("--shuffle-match-dirs", action="store_true",
                   help="Baraja las partidas antes de aplicar --max-matches o sample-frac.")
    p.add_argument("--seed", type=int, default=42,
                   help="Semilla usada para el barajado de partidas.")

    # Jungle
    p.add_argument("--min-frames-used-for-score", type=int, default=4)
    p.add_argument("--exclude-mid-from-active-score", action="store_true")

    # Support
    p.add_argument("--min-support-frames", type=int, default=4)
    p.add_argument("--min-coop-frames", type=int, default=3)

    # Team tendency
    p.add_argument("--min-frames-any-alive", type=int, default=4)
    p.add_argument("--min-frames-with-any-side-signal", type=int, default=3)
    p.add_argument("--min-total-side-mass", type=float, default=3.0)
    p.add_argument("--jg-weight", type=float, default=1.0, help="Peso del jungler en team tendency")
    p.add_argument("--sup-weight", type=float, default=1.0, help="Peso del support en team tendency")
    p.add_argument("--mid-weight", type=float, default=1.0, help="Peso del mid en team tendency")

    # Labeling (compartido por las 3 etiquetas)
    p.add_argument("--labeling-mode", choices=["none", "global", "by_side"], default="by_side")
    p.add_argument("--lower-quantile", type=float, default=0.20)
    p.add_argument("--upper-quantile", type=float, default=0.80)
    p.add_argument("--lower-threshold", type=float, default=None)
    p.add_argument("--upper-threshold", type=float, default=None)
    p.add_argument("--drop-ambiguous", action="store_true")
    p.add_argument(
        "--label-schema",
        choices=["ternary", "binary_clean", "binary_full"],
        default="ternary",
        help=(
            "Esquema de discretización de labels. "
            "ternary = extremo/ambiguous/extremo; "
            "binary_clean = conserva solo extremos y deja el centro como NaN; "
            "binary_full = fuerza todo el espacio en dos clases usando un split intermedio."
        ),
    )

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# CÓMPUTO DE MÉTRICAS POR EQUIPO
# ═══════════════════════════════════════════════════════════════════════════════

def compute_jungle_metrics(
    frames: List[dict], jungle_pid: int, team_id: int,
    include_mid: bool, min_frames: int,
) -> Optional[dict]:
    river_top = river_bot = top = mid = bot = own_jg = enemy_jg = base = other = valid = 0
    for frame in frames:
        pf = get_participant_frame(frame, jungle_pid)
        if not participant_is_alive(pf):
            continue
        pos = extract_position(pf)
        if pos is None:
            continue
        valid += 1
        zone = classify_map_zone(pos[0], pos[1], team_id)
        if zone == "RIVER_TOP":    river_top += 1
        elif zone == "RIVER_BOT":  river_bot += 1
        elif zone == "TOP_LANE":   top += 1
        elif zone == "MID_LANE":   mid += 1
        elif zone == "BOTTOM_LANE": bot += 1
        elif zone in OWN_JUNGLE_ZONES:   own_jg += 1
        elif zone in ENEMY_JUNGLE_ZONES: enemy_jg += 1
        elif zone in BASE_ZONES:         base += 1
        else:                            other += 1

    river = river_top + river_bot
    lane = top + mid + bot
    active_lane = top + bot + (mid if include_mid else 0)
    active_map = river + active_lane + enemy_jg
    frames_used = active_map + own_jg
    if frames_used < min_frames:
        return None
    denom = active_map + own_jg
    score = active_map / denom if denom > 0 else None
    if score is None:
        return None
    coverage = frames_used / valid if valid > 0 else None
    return {
        "valid_jungle_frames": valid,
        "frames_used_for_score": frames_used,
        "score_coverage_ratio": coverage,
        "river_top_frames": river_top, "river_bot_frames": river_bot,
        "river_frames": river, "top_lane_frames": top,
        "mid_lane_frames": mid, "bottom_lane_frames": bot,
        "lane_frames": lane, "active_lane_frames": active_lane,
        "enemy_jungle_frames": enemy_jg, "active_map_frames": active_map,
        "own_jungle_frames": own_jg, "base_frames": base, "other_frames": other,
        "jungle_presence_score": score,
        "active_map_definition": "river+top+mid+bot+enemy_jungle" if include_mid else "river+top+bot+enemy_jungle",
    }


def compute_support_metrics(
    frames: List[dict], support_pid: int, adc_pid: int,
    team_id: int, min_supp: int, min_coop: int,
) -> Optional[dict]:
    """Calcula métricas de support.

    Score = ratio_fuera = frames fuera de bot-side / frames vivos totales.
    Simple, objetivo, sin normalización → sin leakage.
    La distancia media al ADC se mantiene como metadata diagnóstica.
    """
    valid_supp = valid_coop = in_bot = out_bot = 0
    dist_sum = 0.0
    for frame in frames:
        supp_pf = get_participant_frame(frame, support_pid)
        adc_pf = get_participant_frame(frame, adc_pid)
        if not participant_is_alive(supp_pf):
            continue
        pos_s = extract_position(supp_pf)
        if pos_s is None:
            continue
        valid_supp += 1
        zone = classify_map_zone(pos_s[0], pos_s[1], team_id)
        if zone in BOT_SIDE_ZONES:
            in_bot += 1
        else:
            out_bot += 1
        # Distancia al ADC como metadata (no afecta al score)
        if participant_is_alive(adc_pf):
            pos_a = extract_position(adc_pf)
            if pos_a is not None:
                valid_coop += 1
                dist_sum += math.hypot(pos_s[0] - pos_a[0], pos_s[1] - pos_a[1])
    if valid_supp < min_supp or valid_coop < min_coop:
        return None
    ratio_fuera = out_bot / valid_supp if valid_supp > 0 else None
    mean_dist = dist_sum / valid_coop if valid_coop > 0 else None
    coop_ratio = valid_coop / valid_supp if valid_supp > 0 else None
    return {
        "valid_support_frames": valid_supp,
        "valid_coop_frames": valid_coop,
        "coop_frame_ratio": coop_ratio,
        "frames_in_botside": in_bot,
        "frames_out_botside": out_bot,
        "support_roam_score": ratio_fuera,      # ← EL SCORE: puro ratio_fuera
        "mean_distance_to_adc": mean_dist,       # metadata diagnóstica
    }


def compute_team_tendency_metrics(
    frames: List[dict], jg_pid: int, sup_pid: int, mid_pid: int,
    team_id: int,
    w_jg: float, w_sup: float, w_mid: float,
    min_alive: int, min_signal: int, min_mass: float,
) -> Optional[dict]:
    """Calcula la tendencia lateral del equipo.

    Fórmula simplificada sin bonos de coordinación:
        TopPresence_t = w_jg*I_top(JG) + w_sup*I_top(SUP) + w_mid*I_top(MID)
        BotPresence_t = w_jg*I_bot(JG) + w_sup*I_bot(SUP) + w_mid*I_bot(MID)
        Score = (TopScore - BotScore) / (TopScore + BotScore)  ∈ [-1, 1]

    Pesos iguales por defecto (w=1.0). La etiqueta es objetiva;
    las ponderaciones las aprende el modelo.
    """
    valid_alive = signal = top_signal = bot_signal = 0
    top_score_acc = bot_score_acc = 0.0
    for frame in frames:
        zones = {"jg": "NONE", "sup": "NONE", "mid": "NONE"}
        any_alive = False
        for key, pid in (("jg", jg_pid), ("sup", sup_pid), ("mid", mid_pid)):
            pf = get_participant_frame(frame, pid)
            if not participant_is_alive(pf):
                continue
            pos = extract_position(pf)
            if pos is None:
                continue
            any_alive = True
            zones[key] = classify_team_side(pos[0], pos[1], team_id)
        if not any_alive:
            continue
        valid_alive += 1
        tp = (w_jg * int(zones["jg"] == "TOP")
              + w_sup * int(zones["sup"] == "TOP")
              + w_mid * int(zones["mid"] == "TOP"))
        bp = (w_jg * int(zones["jg"] == "BOT")
              + w_sup * int(zones["sup"] == "BOT")
              + w_mid * int(zones["mid"] == "BOT"))
        top_score_acc += tp
        bot_score_acc += bp
        if tp > 0 or bp > 0:
            signal += 1
        if tp > 0:
            top_signal += 1
        if bp > 0:
            bot_signal += 1

    total = top_score_acc + bot_score_acc
    if valid_alive < min_alive or signal < min_signal or total < min_mass:
        return None
    focus = (top_score_acc - bot_score_acc) / total if total > 0 else None
    if focus is None:
        return None
    coverage = signal / valid_alive if valid_alive > 0 else None
    return {
        "valid_frames_any_alive": valid_alive,
        "frames_with_any_side_signal": signal,
        "frames_with_top_signal": top_signal,
        "frames_with_bot_signal": bot_signal,
        "top_score": top_score_acc,
        "bot_score": bot_score_acc,
        "total_side_mass": total,
        "side_signal_coverage_ratio": coverage,
        "role_weights": f"jg={w_jg},sup={w_sup},mid={w_mid}",
        "team_side_focus_score": focus,
    }


def extract_draft_row(
    info: dict, match_id: str, team_id: int,
    role_map: Dict[int, Dict[str, int]], p_lookup: Dict[int, dict],
    bans: Dict[int, List[Optional[int]]],
) -> dict:
    enemy_id = RED_TEAM_ID if team_id == BLUE_TEAM_ID else BLUE_TEAM_ID
    own = role_map[team_id]
    enemy = role_map[enemy_id]
    gv = info.get("gameVersion")
    row: dict = {
        "match_id": match_id, "team_id": team_id,
        "side": side_from_team_id(team_id),
        "patch": infer_patch(gv), "game_version": gv,
        "game_start_timestamp": info.get("gameStartTimestamp"),
        "platform_id": str(info.get("platformId")) if info.get("platformId") else None,
        "queue_id": info.get("queueId") if isinstance(info.get("queueId"), int) else None,
        "game_duration_seconds": safe_game_duration_seconds(info),
    }
    for prefix, roles in (("ally", own), ("enemy", enemy)):
        for canon, lower in zip(CANONICAL_ROLES, ROLE_KEYS_LOWER):
            pid = roles[canon]
            p = p_lookup.get(pid, {})
            row[f"{prefix}_{lower}_participant_id"] = pid
            row[f"{prefix}_{lower}_champion_id"] = p.get("championId")
            row[f"{prefix}_{lower}_champion_name"] = p.get("championName")
            s1, s2 = extract_summoner_spells(p)
            row[f"{prefix}_{lower}_summoner1_id"] = s1
            row[f"{prefix}_{lower}_summoner2_id"] = s2
            ks, prim, sub = extract_runes(p)
            row[f"{prefix}_{lower}_keystone_id"] = ks
            row[f"{prefix}_{lower}_primary_style_id"] = prim
            row[f"{prefix}_{lower}_sub_style_id"] = sub
    own_bans = bans.get(team_id, [None]*5)
    enemy_bans = bans.get(enemy_id, [None]*5)
    for i in range(5):
        row[f"ally_ban_{i+1}_champion_id"] = own_bans[i] if i < len(own_bans) else None
        row[f"enemy_ban_{i+1}_champion_id"] = enemy_bans[i] if i < len(enemy_bans) else None
    return row


# ═══════════════════════════════════════════════════════════════════════════════
# LABELING / DISCRETIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def compute_threshold_pair(
    scores: pd.Series, lower_q: float, upper_q: float,
    lower_thr_arg: Optional[float], upper_thr_arg: Optional[float],
    *, score_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[Optional[float], Optional[float], str]:
    valid = scores.dropna()
    if valid.empty:
        return None, None, "no_valid_scores"
    if (lower_thr_arg is None) ^ (upper_thr_arg is None):
        raise SystemExit("Debes pasar ambos --lower-threshold y --upper-threshold, o ninguno.")
    if lower_thr_arg is not None and upper_thr_arg is not None:
        lo, hi = float(lower_thr_arg), float(upper_thr_arg)
        strategy = "fixed"
    else:
        lo = float(valid.quantile(lower_q))
        hi = float(valid.quantile(upper_q))
        strategy = "quantile"
    if not (score_range[0] <= lo < hi <= score_range[1]):
        raise SystemExit(f"Thresholds deben cumplir {score_range[0]} <= lower < upper <= {score_range[1]}. Got {lo}, {hi}")
    return lo, hi, strategy


def apply_labels_to_df(
    df: pd.DataFrame,
    score_col: str, label_col: str,
    labels_low: str, labels_high: str,
    labeling_mode: str,
    lower_q: float, upper_q: float,
    lower_thr: Optional[float], upper_thr: Optional[float],
    score_range: Tuple[float, float] = (0.0, 1.0),
    label_schema: str = "ternary",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out[label_col] = None
    out["labeling_mode"] = labeling_mode
    out["label_schema"] = label_schema
    out["label_lower_threshold"] = None
    out["label_upper_threshold"] = None
    out["label_split_threshold"] = None

    def _label(score, lo, hi, split_thr):
        if score is None or pd.isna(score):
            return None
        v = float(score)
        if label_schema == "ternary":
            if v <= lo:
                return labels_low
            if v >= hi:
                return labels_high
            return "ambiguous"
        if label_schema == "binary_clean":
            if v <= lo:
                return labels_low
            if v >= hi:
                return labels_high
            return None
        if label_schema == "binary_full":
            return labels_low if v <= split_thr else labels_high
        raise SystemExit(f"label_schema no soportado: {label_schema}")

    meta_rows: List[dict] = []
    if labeling_mode == "none":
        meta_rows.append({
            "group": "ALL", "labeling_mode": "none", "label_schema": label_schema,
            "strategy": "none", "lower": None, "upper": None, "split_threshold": None,
        })
    elif labeling_mode == "global":
        lo, hi, strat = compute_threshold_pair(
            out[score_col], lower_q, upper_q, lower_thr, upper_thr, score_range=score_range
        )
        split_thr = (lo + hi) / 2.0 if lo is not None and hi is not None else None
        if lo is not None and hi is not None:
            out[label_col] = out[score_col].apply(lambda s: _label(s, lo, hi, split_thr))
            out["label_lower_threshold"] = lo
            out["label_upper_threshold"] = hi
            out["label_split_threshold"] = split_thr
        meta_rows.append({
            "group": "ALL", "labeling_mode": "global", "label_schema": label_schema,
            "strategy": strat, "lower": lo, "upper": hi, "split_threshold": split_thr,
        })
    elif labeling_mode == "by_side":
        for side_val, idx in out.groupby("side").groups.items():
            lo, hi, strat = compute_threshold_pair(
                out.loc[idx, score_col], lower_q, upper_q, lower_thr, upper_thr,
                score_range=score_range,
            )
            split_thr = (lo + hi) / 2.0 if lo is not None and hi is not None else None
            if lo is not None and hi is not None:
                out.loc[idx, label_col] = out.loc[idx, score_col].apply(lambda s: _label(s, lo, hi, split_thr))
                out.loc[idx, "label_lower_threshold"] = lo
                out.loc[idx, "label_upper_threshold"] = hi
                out.loc[idx, "label_split_threshold"] = split_thr
            meta_rows.append({
                "group": side_val, "labeling_mode": "by_side", "label_schema": label_schema,
                "strategy": strat, "lower": lo, "upper": hi, "split_threshold": split_thr,
            })
    return out, pd.DataFrame(meta_rows)


# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS / PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def build_percentiles(series: pd.Series, name: str) -> pd.DataFrame:
    valid = series.dropna()
    if valid.empty:
        return pd.DataFrame()
    qs = [0.01, 0.05, 0.10, 0.25, 0.33, 0.50, 0.66, 0.75, 0.90, 0.95, 0.99]
    return pd.DataFrame([{"metric": name, "q": q, "value": float(valid.quantile(q))} for q in qs])


def plot_hist(series: pd.Series, path: str, title: str, xlabel: str) -> None:
    valid = series.dropna()
    if valid.empty:
        return
    ensure_parent_dir(path)
    plt.figure(figsize=(8, 5))
    plt.hist(valid, bins=40)
    plt.xlabel(xlabel); plt.ylabel("count"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def summarize_by_side(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    work = df[df[score_col].notna()]
    if work.empty:
        return pd.DataFrame()
    return work.groupby("side")[score_col].agg(
        n="count", mean="mean", median="median", std="std",
        q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
    ).reset_index().sort_values("side").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    raw_base = os.path.join(args.raw_root, args.region)
    match_dirs = list_match_dirs(raw_base)
    print(f"Directorios de partida detectados: {len(match_dirs)}")
    if args.shuffle_match_dirs:
        rng = random.Random(args.seed)
        rng.shuffle(match_dirs)
        print(f"Partidas barajadas con seed={args.seed}.")

    target_frac = get_target_frac(args.sample_frac)
    suffix = ""
    if target_frac is not None and 0.0 < target_frac < 1.0:
        limit = max(1, int(len(match_dirs) * target_frac))
        match_dirs = match_dirs[:limit]
        suffix = f"_sample{int(target_frac * 100)}"
        print(f"Muestreo ({target_frac}): {limit} partidas.")
    if args.max_matches and args.max_matches > 0:
        match_dirs = match_dirs[:args.max_matches]
        print(f"Limitado a: {len(match_dirs)} partidas.")

    analysis_windows = parse_analysis_windows(args)
    multi_window = args.analysis_max_minutes is not None

    # Rutas de salida
    dr_path = os.path.join(args.out_features_dir, f"{DRAFT_OUT}{suffix}.parquet")
    ensure_parent_dir(dr_path)
    dr_analysis = os.path.splitext(dr_path)[0] + "_analysis"
    ensure_dir(dr_analysis)

    window_output_paths = {m: build_window_output_paths(args.out_labels_dir, suffix, m, multi_window) for m in analysis_windows}
    quantile_tag = format_quantile_or_threshold_tag(
        args.lower_quantile, args.upper_quantile, args.lower_threshold, args.upper_threshold
    )
    for m in analysis_windows:
        for key in ("jungle", "support", "team"):
            window_output_paths[m][key] = apply_quantile_suffix(
                window_output_paths[m][key],
                args.lower_quantile, args.upper_quantile, args.lower_threshold, args.upper_threshold,
            )
        window_output_paths[m]["jungle_analysis"] = os.path.splitext(window_output_paths[m]["jungle"])[0] + "_analysis"
        window_output_paths[m]["support_analysis"] = os.path.splitext(window_output_paths[m]["support"])[0] + "_analysis"
        window_output_paths[m]["team_analysis"] = os.path.splitext(window_output_paths[m]["team"])[0] + "_analysis"
    for paths in window_output_paths.values():
        for key in ("jungle", "support", "team"):
            ensure_parent_dir(paths[key])
        for key in ("jungle_analysis", "support_analysis", "team_analysis"):
            ensure_dir(paths[key])

    print(f"\n[Rutas] RAW: {os.path.abspath(raw_base)}")
    print(f"[Rutas] Labels dir: {os.path.abspath(args.out_labels_dir)}")
    print(f"[Rutas] Features dir: {os.path.abspath(args.out_features_dir)}")
    print(f"[Rutas] Draft features → {os.path.abspath(dr_path)}")
    print(f"[Fase 1] Ventanas de labels: {analysis_windows}")
    print(f"[Fase 2] Tag de quantiles/thresholds: {quantile_tag}")
    print(f"[Fase 2] label_schema: {args.label_schema}")
    for max_minute in analysis_windows:
        paths = window_output_paths[max_minute]
        print(f"  - max_minute={max_minute}: jg={os.path.abspath(paths['jungle'])}")
        print(f"                       sp={os.path.abspath(paths['support'])}")
        print(f"                       tm={os.path.abspath(paths['team'])}")
    print()

    include_mid = not args.exclude_mid_from_active_score

    # Contadores
    total_seen = total_kept = 0
    bad_match = bad_tl = missing_info = short = bad_roles = no_frames = 0
    jg_reject = sp_reject = tm_reject = 0
    missing_team_role = 0

    jg_rows_by_window: DefaultDict[float, List[dict]] = defaultdict(list)
    sp_rows_by_window: DefaultDict[float, List[dict]] = defaultdict(list)
    tm_rows_by_window: DefaultDict[float, List[dict]] = defaultdict(list)
    dr_rows: List[dict] = []

    t0 = time.time()
    last_log = t0

    for mdir in match_dirs:
        total_seen += 1
        if total_seen % 1000 == 0 or (time.time() - last_log) > 15:
            elapsed = time.time() - t0
            rate = total_seen / elapsed if elapsed > 0 else 0
            print(
                f"[{total_seen}/{len(match_dirs)}] kept={total_kept} "
                f"jg={sum(len(v) for v in jg_rows_by_window.values())} sp={sum(len(v) for v in sp_rows_by_window.values())} tm={sum(len(v) for v in tm_rows_by_window.values())} dr={len(dr_rows)} "
                f"rate={rate:.1f}/s"
            )
            last_log = time.time()

        # ── Cargar JSON ──
        match_path = os.path.join(mdir, "match.json")
        tl_path = os.path.join(mdir, "timeline.json")
        try:
            match = load_json(match_path)
        except Exception:
            bad_match += 1; continue
        try:
            timeline = load_json(tl_path)
        except Exception:
            bad_tl += 1; continue

        info = get_match_info(match)
        if not info:
            missing_info += 1; continue
        dur = game_duration_minutes(info)
        if dur is None or dur < args.min_duration_minutes:
            short += 1; continue

        role_map = extract_team_role_map(info)
        if not (BLUE_TEAM_ID in role_map and RED_TEAM_ID in role_map):
            bad_roles += 1; continue

        raw_frames = get_timeline_frames(timeline)
        frames_by_window: Dict[float, List[dict]] = {m: frames_in_window(raw_frames, m) for m in analysis_windows}
        if not any(frames_by_window.values()):
            no_frames += 1; continue

        total_kept += 1
        match_id = get_match_id(match, mdir)
        p_lookup = participant_lookup(info)
        bans = extract_team_bans(info)
        gv = info.get("gameVersion")
        patch = infer_patch(gv)
        game_ts = info.get("gameStartTimestamp") or info.get("gameCreation")

        for team_id in (BLUE_TEAM_ID, RED_TEAM_ID):
            if team_id not in role_map:
                missing_team_role += 1; continue
            rm = role_map[team_id]
            side = side_from_team_id(team_id)
            base_common = {"match_id": match_id, "team_id": team_id, "side": side,
                           "patch": patch, "game_version": gv,
                           "game_start_timestamp": game_ts}

            # Metadata reutilizable por ventana
            jg_pid = rm["JUNGLE"]
            sup_pid = rm["UTILITY"]
            adc_pid = rm["BOTTOM"]
            mid_pid = rm["MIDDLE"]
            jg_meta = p_lookup.get(jg_pid, {})
            sup_meta = p_lookup.get(sup_pid, {})
            adc_meta = p_lookup.get(adc_pid, {})
            mid_meta = p_lookup.get(mid_pid, {})

            for window_max_minute, frames in frames_by_window.items():
                if not frames:
                    continue
                common = dict(base_common)
                common["max_minute"] = window_max_minute
                common["window_tag"] = format_window_tag(window_max_minute)

                # ── Jungle ──
                jg = compute_jungle_metrics(frames, jg_pid, team_id, include_mid, args.min_frames_used_for_score)
                if jg is not None:
                    jg.update(common)
                    jg["jungle_participant_id"] = jg_pid
                    jg["jungle_champion_id"] = jg_meta.get("championId")
                    jg["jungle_champion_name"] = jg_meta.get("championName")
                    jg_rows_by_window[window_max_minute].append(jg)
                else:
                    jg_reject += 1

                # ── Support ──
                sp = compute_support_metrics(frames, sup_pid, adc_pid, team_id,
                                              args.min_support_frames, args.min_coop_frames)
                if sp is not None:
                    sp.update(common)
                    sp["support_participant_id"] = sup_pid
                    sp["support_champion_id"] = sup_meta.get("championId")
                    sp["support_champion_name"] = sup_meta.get("championName")
                    sp["adc_participant_id"] = adc_pid
                    sp["adc_champion_id"] = adc_meta.get("championId")
                    sp["adc_champion_name"] = adc_meta.get("championName")
                    sp_rows_by_window[window_max_minute].append(sp)
                else:
                    sp_reject += 1

                # ── Team tendency ──
                tm = compute_team_tendency_metrics(
                    frames, jg_pid, sup_pid, mid_pid, team_id,
                    args.jg_weight, args.sup_weight, args.mid_weight,
                    args.min_frames_any_alive, args.min_frames_with_any_side_signal,
                    args.min_total_side_mass,
                )
                if tm is not None:
                    tm.update(common)
                    tm["jungle_participant_id"] = jg_pid
                    tm["jungle_champion_id"] = jg_meta.get("championId")
                    tm["jungle_champion_name"] = jg_meta.get("championName")
                    tm["support_participant_id"] = sup_pid
                    tm["support_champion_id"] = sup_meta.get("championId")
                    tm["support_champion_name"] = sup_meta.get("championName")
                    tm["mid_participant_id"] = mid_pid
                    tm["mid_champion_id"] = mid_meta.get("championId")
                    tm["mid_champion_name"] = mid_meta.get("championName")
                    tm_rows_by_window[window_max_minute].append(tm)
                else:
                    tm_reject += 1

            # ── Draft features ──
            dr_rows.append(extract_draft_row(info, match_id, team_id, role_map, p_lookup, bans))

    # ═══════════════════════════════════════════════════════════════════════
    # POST-PROCESAMIENTO
    # ═══════════════════════════════════════════════════════════════════════

    elapsed = time.time() - t0
    print(f"\nProcesado: {total_seen} matches en {elapsed:.1f}s "
          f"({total_seen/elapsed:.1f}/s)" if elapsed > 0 else "")
    print(f"  Kept: {total_kept} | bad_match: {bad_match} | bad_tl: {bad_tl} | "
          f"short: {short} | bad_roles: {bad_roles} | no_frames: {no_frames}")
    print(f"  jg_rows: {sum(len(v) for v in jg_rows_by_window.values())} (reject: {jg_reject}) | "
          f"sp_rows: {sum(len(v) for v in sp_rows_by_window.values())} (reject: {sp_reject}) | "
          f"tm_rows: {sum(len(v) for v in tm_rows_by_window.values())} (reject: {tm_reject}) | "
          f"dr_rows: {len(dr_rows)}")

    # ── Labels por ventana (Fase 1) ──
    window_comparison_rows: List[dict] = []

    for window_max_minute in analysis_windows:
        paths = window_output_paths[window_max_minute]

        # Jungle
        df_jg = pd.DataFrame(jg_rows_by_window.get(window_max_minute, []))
        if not df_jg.empty:
            validate_no_duplicate_keys(df_jg)
            df_jg, jg_meta = apply_labels_to_df(
                df_jg, "jungle_presence_score", "jungle_presence_label",
                "farm_oriented", "map_presence",
                args.labeling_mode, args.lower_quantile, args.upper_quantile,
                args.lower_threshold, args.upper_threshold,
                label_schema=args.label_schema,
            )
            if args.label_schema == "binary_clean":
                df_jg = df_jg[df_jg["jungle_presence_label"].notna()].copy()
            elif args.drop_ambiguous:
                df_jg = df_jg[df_jg["jungle_presence_label"] != "ambiguous"].copy()
            df_jg.sort_values(["match_id", "team_id"]).to_parquet(paths["jungle"], index=False)
            save_dataframe(jg_meta, os.path.join(paths["jungle_analysis"], "threshold_config"))
            save_dataframe(build_percentiles(df_jg["jungle_presence_score"], "jungle_presence_score"),
                           os.path.join(paths["jungle_analysis"], "percentiles"))
            save_dataframe(summarize_by_side(df_jg, "jungle_presence_score"),
                           os.path.join(paths["jungle_analysis"], "side_summary"))
            if not args.skip_plots:
                plot_hist(df_jg["jungle_presence_score"],
                          os.path.join(paths["jungle_analysis"], "score_dist.png"),
                          f"Jungle Presence Score Distribution (0-{window_max_minute:g})", "jungle_presence_score")
            print(f"\n✓ Jungle labels [{window_max_minute:g}]: {len(df_jg)} rows → {paths['jungle']}")
            if df_jg["jungle_presence_label"].notna().any():
                print(df_jg["jungle_presence_label"].value_counts(dropna=False).to_string())
                print(f"  label_schema={args.label_schema}")
                for label_value, count in df_jg["jungle_presence_label"].value_counts(dropna=False).items():
                    window_comparison_rows.append({"task": "jungle", "max_minute": window_max_minute, "label": label_value, "n": int(count)})
        else:
            print(f"\n✗ Jungle labels [{window_max_minute:g}]: 0 rows.")

        # Support
        df_sp = pd.DataFrame(sp_rows_by_window.get(window_max_minute, []))
        if not df_sp.empty:
            validate_no_duplicate_keys(df_sp)
            df_sp["bot_side_definition"] = "BOTTOM_LANE|OWN_BOTTOM_JUNGLE"
            df_sp, sp_meta = apply_labels_to_df(
                df_sp, "support_roam_score", "support_roam_label",
                "lane_anchored", "roamer",
                args.labeling_mode, args.lower_quantile, args.upper_quantile,
                args.lower_threshold, args.upper_threshold,
                label_schema=args.label_schema,
            )
            if args.label_schema == "binary_clean":
                df_sp = df_sp[df_sp["support_roam_label"].notna()].copy()
            elif args.drop_ambiguous:
                df_sp = df_sp[df_sp["support_roam_label"] != "ambiguous"].copy()
            df_sp.sort_values(["match_id", "team_id"]).to_parquet(paths["support"], index=False)
            save_dataframe(sp_meta, os.path.join(paths["support_analysis"], "threshold_config"))
            save_dataframe(build_percentiles(df_sp["support_roam_score"], "support_roam_score"),
                           os.path.join(paths["support_analysis"], "percentiles"))
            save_dataframe(summarize_by_side(df_sp, "support_roam_score"),
                           os.path.join(paths["support_analysis"], "side_summary"))
            if not args.skip_plots:
                plot_hist(df_sp["support_roam_score"],
                          os.path.join(paths["support_analysis"], "score_dist.png"),
                          f"Support Roam Score Distribution (0-{window_max_minute:g})", "support_roam_score")
            print(f"\n✓ Support labels [{window_max_minute:g}]: {len(df_sp)} rows → {paths['support']}")
            print("  Score = ratio_fuera (sin normalización, sin leakage)")
            if df_sp["support_roam_label"].notna().any():
                print(df_sp["support_roam_label"].value_counts(dropna=False).to_string())
                print(f"  label_schema={args.label_schema}")
                for label_value, count in df_sp["support_roam_label"].value_counts(dropna=False).items():
                    window_comparison_rows.append({"task": "support", "max_minute": window_max_minute, "label": label_value, "n": int(count)})
        else:
            print(f"\n✗ Support labels [{window_max_minute:g}]: 0 rows.")

        # Team
        df_tm = pd.DataFrame(tm_rows_by_window.get(window_max_minute, []))
        if not df_tm.empty:
            validate_no_duplicate_keys(df_tm)
            df_tm, tm_meta = apply_labels_to_df(
                df_tm, "team_side_focus_score", "team_tendency_label",
                "botside_oriented", "topside_oriented",
                args.labeling_mode, args.lower_quantile, args.upper_quantile,
                args.lower_threshold, args.upper_threshold,
                score_range=(-1.0, 1.0),
                label_schema=args.label_schema,
            )
            if args.label_schema == "binary_clean":
                df_tm = df_tm[df_tm["team_tendency_label"].notna()].copy()
            elif args.drop_ambiguous:
                df_tm = df_tm[df_tm["team_tendency_label"] != "ambiguous"].copy()
            df_tm.sort_values(["match_id", "team_id"]).to_parquet(paths["team"], index=False)
            save_dataframe(tm_meta, os.path.join(paths["team_analysis"], "threshold_config"))
            save_dataframe(build_percentiles(df_tm["team_side_focus_score"], "team_side_focus_score"),
                           os.path.join(paths["team_analysis"], "percentiles"))
            save_dataframe(summarize_by_side(df_tm, "team_side_focus_score"),
                           os.path.join(paths["team_analysis"], "side_summary"))
            if not args.skip_plots:
                plot_hist(df_tm["team_side_focus_score"],
                          os.path.join(paths["team_analysis"], "score_dist.png"),
                          f"Team Side Focus Score Distribution (0-{window_max_minute:g})", "team_side_focus_score")
            print(f"\n✓ Team tendency labels [{window_max_minute:g}]: {len(df_tm)} rows → {paths['team']}")
            if df_tm["team_tendency_label"].notna().any():
                print(df_tm["team_tendency_label"].value_counts(dropna=False).to_string())
                print(f"  label_schema={args.label_schema}")
                for label_value, count in df_tm["team_tendency_label"].value_counts(dropna=False).items():
                    window_comparison_rows.append({"task": "team", "max_minute": window_max_minute, "label": label_value, "n": int(count)})
        else:
            print(f"\n✗ Team tendency labels [{window_max_minute:g}]: 0 rows.")

    # ── Draft features ──
    df_dr = pd.DataFrame(dr_rows)
    if not df_dr.empty:
        validate_no_duplicate_keys(df_dr)
        df_dr.sort_values(["match_id", "team_id"]).to_parquet(dr_path, index=False)
        save_dataframe(
            pd.DataFrame([{"rows": len(df_dr),
                           "matches": int(df_dr["match_id"].nunique()),
                           "cols": len(df_dr.columns)}]),
            os.path.join(dr_analysis, "overall_summary"),
        )
        if window_comparison_rows:
            save_dataframe(pd.DataFrame(window_comparison_rows).sort_values(["task", "max_minute", "label"]),
                           os.path.join(dr_analysis, "window_label_distribution_summary"))
        save_dataframe(
            pd.DataFrame([{
                "raw_base": os.path.abspath(raw_base),
                "out_labels_dir": os.path.abspath(args.out_labels_dir),
                "out_features_dir": os.path.abspath(args.out_features_dir),
                "draft_features_path": os.path.abspath(dr_path),
                "label_schema": args.label_schema,
                "labeling_mode": args.labeling_mode,
                "quantile_or_threshold_tag": quantile_tag,
                "lower_quantile": args.lower_quantile,
                "upper_quantile": args.upper_quantile,
                "lower_threshold": args.lower_threshold,
                "upper_threshold": args.upper_threshold,
                "analysis_windows": ",".join(str(x) for x in analysis_windows),
                "windows_count": len(analysis_windows),
                "multi_window": bool(multi_window),
                "sample_frac": target_frac,
                "drop_ambiguous_flag": bool(args.drop_ambiguous),
                "min_duration_minutes": args.min_duration_minutes,
                "max_matches": args.max_matches,
                "shuffle_match_dirs": bool(args.shuffle_match_dirs),
                "seed": args.seed,
                "include_mid_in_jungle_active_score": bool(include_mid),
                "min_frames_used_for_score": args.min_frames_used_for_score,
                "min_support_frames": args.min_support_frames,
                "min_coop_frames": args.min_coop_frames,
                "min_frames_any_alive": args.min_frames_any_alive,
                "min_frames_with_any_side_signal": args.min_frames_with_any_side_signal,
                "min_total_side_mass": args.min_total_side_mass,
                "jg_weight": args.jg_weight,
                "sup_weight": args.sup_weight,
                "mid_weight": args.mid_weight,
                "skip_plots": bool(args.skip_plots),
                "matches_seen": total_seen,
                "matches_kept": total_kept,
                "draft_rows": len(df_dr),
                "jungle_rows_total": sum(len(v) for v in jg_rows_by_window.values()),
                "support_rows_total": sum(len(v) for v in sp_rows_by_window.values()),
                "team_rows_total": sum(len(v) for v in tm_rows_by_window.values()),
            }]),
            os.path.join(dr_analysis, "label_build_run_config"),
        )
        save_dataframe(
            pd.DataFrame([{
                "overall_summary": os.path.join(dr_analysis, "overall_summary"),
                "window_label_distribution_summary": os.path.join(dr_analysis, "window_label_distribution_summary") if window_comparison_rows else None,
                "label_build_run_config": os.path.join(dr_analysis, "label_build_run_config"),
            }]),
            os.path.join(dr_analysis, "artifact_manifest"),
        )
        print(f"\n✓ Draft features: {len(df_dr)} rows, {len(df_dr.columns)} cols → {dr_path}")
        print("  Incluye: campeones, hechizos de invocador, runas (keystone + estilos), bans")
        if window_comparison_rows:
            print(f"  Añadido resumen comparativo de ventanas → {os.path.join(dr_analysis, 'window_label_distribution_summary')}")
        print(f"  Config de ejecución guardada en → {os.path.join(dr_analysis, 'label_build_run_config')}")
    else:
        print("\n✗ Draft features: 0 rows.")


if __name__ == "__main__":
    main()