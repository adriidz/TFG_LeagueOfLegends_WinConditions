import os
import json
import csv
import glob
import math
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

@dataclass(frozen=True)
class Config:
    raw_root: str = os.getenv("RAW_ROOT", "Data_raw/raw")
    routing_region: str = os.getenv("REGION", "europe")
    out_csv: str = os.getenv("LABELS_OUT", "Data_clean/ds_win.csv")

    # Ventanas
    early_start: int = 0
    early_end: int = 14
    mid_start: int = 14
    mid_end: int = 25

    # Neutralidad (filtro / máscara)
    neutral_minute: int = 14
    neutral_gold_thresh: int = 2500
    neutral_tower_thresh: int = 1

    # Heurísticas OBS_*
    focus_ratio_hi: float = 0.58
    focus_ratio_lo: float = 0.42
    jg_gank_hi: float = 0.35
    jg_farm_lo: float = 0.15
    group_radius: float = 3200.0
    group_ratio_hi: float = 0.20
    split_ratio_hi: float = 0.60  # en v1 lo usamos como "no-group" proxy

    # Parche major.minor (ej: 16.2)
    min_patch_major: int = int(os.getenv("MIN_PATCH_MAJOR", "16"))
    min_patch_minor: int = int(os.getenv("MIN_PATCH_MINOR", "2"))

    # --- filtros de dataset para entrenar ---
    filter_patch: bool = (os.getenv("FILTER_PATCH", "1") == "1")
    filter_neutral_only: bool = (os.getenv("FILTER_NEUTRAL_ONLY", "1") == "1")
    filter_win_only: bool = (os.getenv("FILTER_WIN_ONLY", "1") == "1")

    # --- columnas opcionales (EDA) ---
    include_success_cols: bool = (os.getenv("INCLUDE_SUCCESS_COLS", "0") == "1")


def iter_match_dirs(cfg: Config):
    base = os.path.join(cfg.raw_root, cfg.routing_region)
    for d in glob.glob(os.path.join(base, "*")):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "match.json")) and os.path.exists(os.path.join(d, "timeline.json")):
            yield d

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_frames(tl: dict) -> List[dict]:
    return ((tl.get("info") or {}).get("frames") or [])

def safe_participants(match: dict) -> List[dict]:
    return (((match.get("info") or {}).get("participants") or []))

def participant_team_map(participants: List[dict]) -> Dict[int, int]:
    m: Dict[int, int] = {}
    for p in participants:
        pid = p.get("participantId")
        tid = p.get("teamId")
        if pid is None or tid is None:
            continue
        m[int(pid)] = int(tid)
    return m

def find_pid_by_role(participants: List[dict], team_id: int, role: str) -> Optional[int]:
    for p in participants:
        if p.get("teamId") == team_id and p.get("teamPosition") == role:
            pid = p.get("participantId")
            return int(pid) if pid is not None else None
    return None

def get_draft_features(participants: List[dict], team_id: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in participants:
        if p.get("teamId") != team_id:
            continue
        role = p.get("teamPosition") or "UNKNOWN"
        out[f"X_{role}_Champ"] = p.get("championId")
    return out

def get_frame_at_minute(frames: List[dict], minute: int) -> Optional[dict]:
    if not frames:
        return None
    idx = min(max(0, minute), len(frames) - 1)
    return frames[idx]

def team_total_gold(frame: dict, team_id: int, pid_to_team: Dict[int, int]) -> int:
    pframes = (frame.get("participantFrames") or {})
    total = 0
    for pid_str, pf in pframes.items():
        try:
            pid = int(pid_str)
        except Exception:
            continue
        if pid_to_team.get(pid) == team_id:
            total += int(pf.get("totalGold", 0))
    return total

def tower_diff_until_ms(frames: List[dict], t_ms: int) -> int:
    # blue_taken - red_taken
    diff = 0
    for fr in frames:
        for ev in (fr.get("events") or []):
            if ev.get("timestamp", 10**18) > t_ms:
                continue
            if ev.get("type") != "BUILDING_KILL":
                continue
            if ev.get("buildingType") != "TOWER_BUILDING":
                continue
            destroyed_team = ev.get("teamId")
            if destroyed_team == 100:
                diff -= 1
            elif destroyed_team == 200:
                diff += 1
    return diff

def get_map_side_simple(x: Optional[float], y: Optional[float]) -> str:
    if x is None or y is None:
        return "NEUTRAL"
    return "TOPSIDE" if y > x else "BOTSIDE"

def get_map_side_robust(x: Optional[float], y: Optional[float], neutral_corridor: float = 1500.0) -> str:
    """
    Determina en qué lado del mapa está el jugador, ignorando las bases 
    y estableciendo un pasillo neutral en la diagonal central (Mid Lane).
    """
    if x is None or y is None:
        return "NEUTRAL"
        
    # 1. Zonas muertas: Bases (Fountain / Nexo)
    # Si están en base comprando o reviviendo, no ejercen presión en el mapa.
    # Base Azul (Abajo Izquierda) y Base Roja (Arriba Derecha)
    if (x < 2500 and y < 2500) or (x > 12300 and y > 12300):
        return "NEUTRAL" # O puedes devolver "BASE" si quieres trackearlo aparte
        
    # 2. Corredor Neutral: La diagonal central
    # Aplicamos |y - x| < umbral para crear un "pasillo" alrededor de Mid/Rio
    if abs(y - x) < neutral_corridor:
        return "NEUTRAL"
        
    # 3. Clasificación final
    return "TOPSIDE" if y > x else "BOTSIDE"

def is_in_lane_zone(x: Optional[float], y: Optional[float]) -> bool:
    if x is None or y is None:
        return False
    if y > 11000 and x < 4000:
        return True
    if y < 4000 and x > 11000:
        return True
    if abs(x - y) < 2000 and 3000 < x < 12000:
        return True
    return False

def euclidean_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)

def parse_patch_mm(game_version: Optional[str]) -> Optional[Tuple[int, int]]:
    if not game_version:
        return None
    try:
        parts = str(game_version).split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return None

def is_patch_at_least(mm: Optional[Tuple[int, int]], min_major: int, min_minor: int) -> bool:
    if mm is None:
        return False
    major, minor = mm
    return (major > min_major) or (major == min_major and minor >= min_minor)

# --------- (EDA opcional) MacroEvents + SUCCESS_* ----------
@dataclass
class MacroEvents:
    grubs: int = 0
    heralds: int = 0
    dragons: int = 0
    barons: int = 0
    towers_top: int = 0
    towers_mid: int = 0
    towers_bot: int = 0
    plates: int = 0

def extract_macro_events(frames: List[dict], team_id: int, end_min: int) -> MacroEvents:
    end_ms = end_min * 60_000
    out = MacroEvents()
    for fr in frames:
        for ev in (fr.get("events") or []):
            ts = ev.get("timestamp", 10**18)
            if ts > end_ms:
                continue

            et = ev.get("type")
            if et == "ELITE_MONSTER_KILL":
                if ev.get("killerTeamId") != team_id:
                    continue
                mon = ev.get("monsterType")
                if mon == "HORDE":
                    out.grubs += 1
                elif mon == "RIFTHERALD":
                    out.heralds += 1
                elif mon == "DRAGON":
                    out.dragons += 1
                elif mon == "BARON_NASHOR":
                    out.barons += 1

            elif et == "BUILDING_KILL" and ev.get("buildingType") == "TOWER_BUILDING":
                destroyed_team = ev.get("teamId")
                if (destroyed_team == 100 and team_id == 200) or (destroyed_team == 200 and team_id == 100):
                    lane = ev.get("laneType", "")
                    if lane == "TOP_LANE":
                        out.towers_top += 1
                    elif lane == "MID_LANE":
                        out.towers_mid += 1
                    elif lane == "BOT_LANE":
                        out.towers_bot += 1

            elif et == "TURRET_PLATE_DESTROYED":
                killer_team = ev.get("killerTeamId")
                if killer_team is not None:
                    if killer_team == team_id:
                        out.plates += 1
                else:
                    destroyed_team = ev.get("teamId")
                    if (destroyed_team == 100 and team_id == 200) or (destroyed_team == 200 and team_id == 100):
                        out.plates += 1
    return out

def subtract_events(a: MacroEvents, b: MacroEvents) -> MacroEvents:
    return MacroEvents(
        grubs=a.grubs - b.grubs,
        heralds=a.heralds - b.heralds,
        dragons=a.dragons - b.dragons,
        barons=a.barons - b.barons,
        towers_top=a.towers_top - b.towers_top,
        towers_mid=a.towers_mid - b.towers_mid,
        towers_bot=a.towers_bot - b.towers_bot,
        plates=a.plates - b.plates,
    )

def obs_team_focus_early(frames, team_id, participants, cfg):
    jg_id = find_pid_by_role(participants, team_id, "JUNGLE")
    if jg_id is None:
        return "NEUTRAL"

    top_frames = 0
    bot_frames = 0
    limit = min(cfg.early_end + 1, len(frames))

    for i in range(1, limit):
        pframes = frames[i].get("participantFrames") or {}
        pf = pframes.get(str(jg_id))
        if not pf:
            continue
        pos = pf.get("position") or {}
        side = get_map_side_robust(pos.get("x"), pos.get("y"))
        if side == "TOPSIDE":
            top_frames += 1
        elif side == "BOTSIDE":
            bot_frames += 1

    total = top_frames + bot_frames
    if total == 0:
        return "NEUTRAL"

    top_ratio = top_frames / total
    if top_ratio > cfg.focus_ratio_hi:
        return "TOP"
    if top_ratio < cfg.focus_ratio_lo:
        return "BOT"
    return "NEUTRAL"

def success_team_focus_early(focus: str, ev: MacroEvents) -> Optional[int]:
    if focus == "TOP":
        return 1 if (ev.heralds >= 1 or ev.grubs >= 3 or ev.towers_top >= 1) else 0
    if focus == "BOT":
        return 1 if (ev.dragons >= 1 or ev.towers_bot >= 1) else 0
    return None

def obs_jg_style_early(frames: List[dict], team_id: int, participants: List[dict], cfg: Config) -> str:
    jg_id = find_pid_by_role(participants, team_id, "JUNGLE")
    if jg_id is None:
        return "MIXED"

    total = 0
    lane = 0
    limit = min(cfg.early_end + 1, len(frames))

    for i in range(1, limit):
        pframes = frames[i].get("participantFrames") or {}
        pf = pframes.get(str(jg_id))
        if not pf:
            continue
        total += 1
        pos = pf.get("position") or {}
        if is_in_lane_zone(pos.get("x"), pos.get("y")):
            lane += 1

    if total == 0:
        return "MIXED"

    ratio = lane / total
    if ratio > cfg.jg_gank_hi:
        return "GANK_HEAVY"
    if ratio < cfg.jg_farm_lo:
        return "FARM_HEAVY"
    return "MIXED"

def success_jg_style_early(style: str, ev: MacroEvents) -> Optional[int]:
    total_objs = ev.dragons + ev.heralds + ev.grubs
    total_towers = ev.towers_top + ev.towers_mid + ev.towers_bot
    if style == "GANK_HEAVY":
        return 1 if (total_objs >= 1 or total_towers >= 1 or ev.plates >= 3) else 0
    if style == "FARM_HEAVY":
        return 1 if (total_objs >= 2) else 0
    return None

def obs_map_mode_mid(frames: List[dict], team_id: int, participants: List[dict], cfg: Config) -> str:
    team_pids = [str(p["participantId"]) for p in participants if p.get("teamId") == team_id and p.get("participantId") is not None]
    if not team_pids:
        return "NEUTRAL"

    start_idx = min(cfg.mid_start, len(frames) - 1)
    end_idx = min(cfg.mid_end + 1, len(frames))

    group_frames = 0
    total_frames = 0

    for i in range(start_idx, end_idx):
        pframes = frames[i].get("participantFrames") or {}
        positions: List[Tuple[float, float]] = []
        for pid in team_pids:
            pf = pframes.get(pid)
            if not pf:
                continue
            pos = pf.get("position")
            if not pos:
                continue
            x, y = pos.get("x"), pos.get("y")
            if x is None or y is None:
                continue
            positions.append((float(x), float(y)))

        if len(positions) < 4:
            continue

        total_frames += 1
        cx = sum(x for x, _ in positions) / len(positions)
        cy = sum(y for _, y in positions) / len(positions)
        grouped = sum(1 for x, y in positions if euclidean_xy(x, y, cx, cy) < cfg.group_radius)

        if grouped >= 4:
            group_frames += 1

    if total_frames == 0:
        return "NEUTRAL"

    group_ratio = group_frames / total_frames
    split_ratio = 1.0 - group_ratio

    if group_ratio > cfg.group_ratio_hi:
        return "GROUP"
    if split_ratio > cfg.split_ratio_hi:
        return "SPLIT"
    return "NEUTRAL"

def success_map_mode_mid(mode: str, ev: MacroEvents) -> Optional[int]:
    if mode == "GROUP":
        return 1 if (ev.towers_mid >= 1 or ev.dragons >= 1 or ev.barons >= 1) else 0
    if mode == "SPLIT":
        return 1 if (ev.towers_top >= 1 or ev.towers_bot >= 1) else 0
    return None

def get_team_win_map(match_info: dict) -> Dict[int, int]:
    """Devuelve {100:0/1, 200:0/1}."""
    teams = (match_info.get("teams") or [])
    out = {100: 0, 200: 0}
    for t in teams:
        tid = t.get("teamId")
        if tid in (100, 200):
            out[tid] = 1 if t.get("win") else 0
    return out

def process(cfg: Config):
    os.makedirs(os.path.dirname(cfg.out_csv) or ".", exist_ok=True)

    t0 = time.perf_counter()

    raw_dirs_total = 0
    raw_dirs_valid_json = 0
    raw_dirs_pass_patch = 0

    # --- contadores de embudo / skips ---
    skip_bad_json = 0
    skip_no_frames = 0
    skip_patch = 0
    skip_too_short_early = 0
    skip_no_frame_14 = 0
    skip_neutral = 0
    skip_win = 0

    rows_total = 0
    rows_written = 0

    rows: List[Dict[str, Any]] = []
    draft_keys: Optional[List[str]] = None

    for md in iter_match_dirs(cfg):
        raw_dirs_total += 1

        if raw_dirs_total % 1000 == 0:
            elapsed = time.perf_counter() - t0
            rate = raw_dirs_total / elapsed if elapsed > 0 else 0.0
            print(f"[PROGRESS] {raw_dirs_total} carpetas | {elapsed:.1f}s | {rate:.1f} carpetas/s")

        try:
            match = load_json(os.path.join(md, "match.json"))
            tl = load_json(os.path.join(md, "timeline.json"))
        except Exception:
            skip_bad_json += 1
            continue

        raw_dirs_valid_json += 1

        info = match.get("info") or {}
        participants = safe_participants(match)
        frames = safe_frames(tl)
        if not frames:
            skip_no_frames += 1
            continue

        # filtro parche
        mm = parse_patch_mm(info.get("gameVersion"))
        pass_patch = is_patch_at_least(mm, cfg.min_patch_major, cfg.min_patch_minor)
        if pass_patch:
            raw_dirs_pass_patch += 1
        if cfg.filter_patch and (not pass_patch):
            skip_patch += 1
            continue

        game_duration = info.get("gameDuration") or 0
        if game_duration < (cfg.early_end + 1) * 60:
            skip_too_short_early += 1
            continue

        pid_to_team = participant_team_map(participants)

        fr14 = get_frame_at_minute(frames, cfg.neutral_minute)
        if not fr14:
            skip_no_frame_14 += 1
            continue

        blue_gold = team_total_gold(fr14, 100, pid_to_team)
        red_gold = team_total_gold(fr14, 200, pid_to_team)
        gold_diff_14 = blue_gold - red_gold
        tower_diff_14 = tower_diff_until_ms(frames, cfg.neutral_minute * 60_000)

        is_neutral_early = int(
            abs(gold_diff_14) <= cfg.neutral_gold_thresh and
            abs(tower_diff_14) <= cfg.neutral_tower_thresh
        )
        if cfg.filter_neutral_only and is_neutral_early != 1:
            skip_neutral += 1
            continue

        patch = f"{mm[0]}.{mm[1]}" if mm else None
        match_id = info.get("gameId") or os.path.basename(md)

        # win por team (NUEVO)
        win_map = get_team_win_map(info)

        # eventos (solo si vas a incluir SUCCESS_*)
        has_mid = game_duration >= (cfg.mid_end + 1) * 60
        if cfg.include_success_cols:
            ev_0_14_100 = extract_macro_events(frames, 100, cfg.early_end)
            ev_0_14_200 = extract_macro_events(frames, 200, cfg.early_end)

            if has_mid:
                ev_0_25_100 = extract_macro_events(frames, 100, cfg.mid_end)
                ev_0_25_200 = extract_macro_events(frames, 200, cfg.mid_end)
                ev_14_25_100 = subtract_events(ev_0_25_100, ev_0_14_100)
                ev_14_25_200 = subtract_events(ev_0_25_200, ev_0_14_200)
            else:
                ev_14_25_100 = MacroEvents()
                ev_14_25_200 = MacroEvents()
        else:
            ev_0_14_100 = ev_0_14_200 = None
            ev_14_25_100 = ev_14_25_200 = None

        for team_id in (100, 200):
            did_win = int(win_map.get(team_id, 0))
            if cfg.filter_win_only and did_win != 1:
                skip_win += 1
                continue

            row: Dict[str, Any] = {
                "matchId": match_id,
                "teamId": team_id,
                "win": did_win,
                "patch": patch,
                "window_early": f"{cfg.early_start}-{cfg.early_end}",
                "window_mid": f"{cfg.mid_start}-{cfg.mid_end}",
                "is_neutral_early": is_neutral_early,
                "gold_diff_14": gold_diff_14,
                "tower_diff_14": tower_diff_14,
            }

            draft = get_draft_features(participants, team_id)
            if draft_keys is None:
                draft_keys = sorted(draft.keys())
            row.update(draft)

            # ---- SOLO 3 OBS_* (core) ----
            focus = obs_team_focus_early(frames, team_id, participants, cfg)
            row["OBS_TEAM_FOCUS_EARLY"] = focus

            jg = obs_jg_style_early(frames, team_id, participants, cfg)
            row["OBS_JG_STYLE_EARLY"] = jg

            if has_mid:
                mmode = obs_map_mode_mid(frames, team_id, participants, cfg)
                row["OBS_MAP_MODE_MID"] = mmode
            else:
                row["OBS_MAP_MODE_MID"] = None

            # ---- SUCCESS_* opcional (EDA) ----
            if cfg.include_success_cols:
                ev_0_14 = ev_0_14_100 if team_id == 100 else ev_0_14_200
                ev_14_25 = ev_14_25_100 if team_id == 100 else ev_14_25_200

                row["SUCCESS_TEAM_FOCUS_EARLY"] = success_team_focus_early(focus, ev_0_14)
                row["SUCCESS_JG_STYLE_EARLY"] = success_jg_style_early(jg, ev_0_14)
                row["SUCCESS_MAP_MODE_MID"] = success_map_mode_mid(row["OBS_MAP_MODE_MID"], ev_14_25) if has_mid else None

            rows.append(row)
            rows_total += 1

    # -----------------------------
    # Escribir CSV (una sola vez)
    # -----------------------------
    base_cols = [
        "matchId", "teamId", "win", "patch",
        "window_early", "window_mid",
        "is_neutral_early", "gold_diff_14", "tower_diff_14",
        "OBS_TEAM_FOCUS_EARLY",
        "OBS_JG_STYLE_EARLY",
        "OBS_MAP_MODE_MID",
    ]

    if cfg.include_success_cols:
        base_cols += [
            "SUCCESS_TEAM_FOCUS_EARLY",
            "SUCCESS_JG_STYLE_EARLY",
            "SUCCESS_MAP_MODE_MID",
        ]

    draft_cols = draft_keys or []
    cols = base_cols + draft_cols

    with open(cfg.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
            rows_written += 1

    # -----------------------------
    # Reporte
    # -----------------------------
    def pct(a: int, b: int) -> float:
        return 0.0 if b == 0 else (100.0 * a / b)

    print("\n=== CONTADORES RAW / FILTROS ===")
    print(f"RAW matches (carpetas) encontrados:          {raw_dirs_total}")
    print(f"RAW matches leíbles (JSON ok):              {raw_dirs_valid_json} ({pct(raw_dirs_valid_json, raw_dirs_total):.1f}%)")
    print(f"RAW matches parche >= {cfg.min_patch_major}.{cfg.min_patch_minor}: {raw_dirs_pass_patch} ({pct(raw_dirs_pass_patch, raw_dirs_valid_json):.1f}% de leíbles)")

    print("\n=== EMBUDO DE FILTROS (skips) ===")
    print(f"skip_bad_json:                           {skip_bad_json}")
    print(f"skip_no_frames:                          {skip_no_frames}")
    print(f"skip_patch (<{cfg.min_patch_major}.{cfg.min_patch_minor}):              {skip_patch}")
    print(f"skip_too_short_early (<{cfg.early_end+1}min):             {skip_too_short_early}")
    print(f"skip_no_frame_at_{cfg.neutral_minute}min:                 {skip_no_frame_14}")
    print(f"skip_not_neutral_early:                  {skip_neutral}")
    print(f"skip_win_only (filas/equipos):           {skip_win}")

    print("\n=== FILAS ===")
    print(f"Filas candidatas (tras filtros):            {rows_total}")
    print(f"Filas escritas a CSV:                       {rows_written}")
    print(f"\nCSV: {cfg.out_csv}")
    print(f"¡Éxito! CSV generado con {rows_written} filas en {cfg.out_csv}")

    if rows:
        def dist(key: str):
            c = Counter(r.get(key) for r in rows)
            total = sum(c.values()) or 1
            print(f"\n{key}:")
            for k, v in c.most_common():
                print(f"  {k}: {v/total:.3f} ({v})")

        print("\n--- REPORTE (dataset final) ---")
        print(f"Muestras: {len(rows)}")
        dist("OBS_TEAM_FOCUS_EARLY")
        dist("OBS_JG_STYLE_EARLY")
        dist("OBS_MAP_MODE_MID")

    dt = time.perf_counter() - t0
    per_dir = (dt / raw_dirs_total) if raw_dirs_total else 0.0
    per_row = (dt / rows_written) if rows_written else 0.0

    print("\n=== TIEMPOS ===")
    print(f"Tiempo total:                         {dt:.2f} s")
    print(f"Tiempo medio por carpeta RAW:         {per_dir*1000:.2f} ms")
    print(f"Tiempo medio por fila escrita:        {per_row*1000:.2f} ms")


if __name__ == "__main__":
    process(Config())