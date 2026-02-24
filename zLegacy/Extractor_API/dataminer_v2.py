import os
import math
import time
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
from riotwatcher import LolWatcher, RiotWatcher, ApiError

# --- CONFIGURACIÓN ---
load_dotenv('TFG.env')

API_KEY = os.getenv('RIOT_API_KEY')
REGION = os.getenv('REGION', 'europe')
MATCH_REGION = os.getenv('MATCH_REGION', 'euw1')

GAME_NAME = 'adriidz'
TAG_LINE = 'diaz'

if not API_KEY:
    raise ValueError("ERROR: No se encontró la variable 'RIOT_API_KEY' en el archivo .env")

# Inicializamos RiotWatcher
watcher = LolWatcher(API_KEY)
account_watcher = RiotWatcher(API_KEY)


def get_match_ids(puuid, count=5, ranked_only=False):
    """Obtiene los IDs usando RiotWatcher (maneja 429 automáticamente)."""
    queue = 420 if ranked_only else None
    try:
        return watcher.match.matchlist_by_puuid(REGION, puuid, count=count, queue=queue)
    except ApiError as err:
        print(f"Error obteniendo IDs: {err}")
        return []


# -------------------------
# POSICIÓN / CLUSTER
# -------------------------

def _euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _safe_pos(obj):
    """obj puede ser participantFrame o event; retorna (x,y) o None."""
    pos = obj.get('position')
    if not pos:
        return None
    x = pos.get('x')
    y = pos.get('y')
    if x is None or y is None:
        return None
    return (float(x), float(y))


def _lane_by_position_xy(x, y):
    """Clasificador simple (pero bastante robusto) de lane por coordenadas."""
    # Mid: cerca de diagonal / centro
    if abs(x - y) <= 2000 and 3500 <= x <= 11500 and 3500 <= y <= 11500:
        return 'MID'
    # Top: arriba-izquierda
    if y >= 9000 and x <= 9000:
        return 'TOP'
    # Bot: abajo-derecha
    if x >= 6000 and y <= 6000:
        return 'BOT'

    # Fallback por distancia a centros
    centers = {'TOP': (2500, 12500), 'MID': (7500, 7500), 'BOT': (12500, 2500)}
    d = {k: _euclid((x, y), v) for k, v in centers.items()}
    return min(d, key=d.get)


def get_positions_by_minute(timeline, minutes_to_track=(5, 10, 15, 20)):
    """positions[minute][participantId] = (x,y)"""
    frames = timeline['info']['frames']
    positions = {m: {} for m in minutes_to_track}
    max_i = len(frames) - 1

    for m in minutes_to_track:
        idx = min(m, max_i)
        frame = frames[idx]
        for pid_str, pframe in frame['participantFrames'].items():
            pid = int(pid_str)
            positions[m][pid] = _safe_pos(pframe)
    return positions


def compute_team_cluster_metrics(positions_at_minute, participants_info, team_id, grouped_radius=2200.0):
    """Spread medio al centroide + ratio de jugadores dentro del radio."""
    pts = []
    for p in participants_info:
        if p['teamId'] != team_id:
            continue
        pid = p['participantId']
        pos = positions_at_minute.get(pid)
        if pos is not None:
            pts.append(pos)

    if len(pts) < 3:
        return {'spread_mean': None, 'grouped_ratio': None}

    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    centroid = (cx, cy)

    dists = [_euclid(pt, centroid) for pt in pts]
    spread_mean = sum(dists) / len(dists)
    grouped_ratio = sum(1 for d in dists if d <= grouped_radius) / len(dists)

    return {'spread_mean': spread_mean, 'grouped_ratio': grouped_ratio}


def compute_jungle_lane_presence(timeline, participants_info, team_id, start_min=2, end_min=14):
    """Presencia del jungla por lane entre start_min..end_min (frames ~ min)."""
    jg_pid = None
    for p in participants_info:
        if p['teamId'] == team_id and p.get('teamPosition') == 'JUNGLE':
            jg_pid = p['participantId']
            break

    if jg_pid is None:
        return {
            'jg_presence_top': None,
            'jg_presence_mid': None,
            'jg_presence_bot': None,
            'jg_focus_lane': None,
        }

    frames = timeline['info']['frames']
    max_i = len(frames) - 1
    s = max(0, min(start_min, max_i))
    e = max(0, min(end_min, max_i))

    counts = {'TOP': 0, 'MID': 0, 'BOT': 0}
    total = 0

    for i in range(s, e + 1):
        pf = frames[i]['participantFrames'].get(str(jg_pid))
        if not pf:
            continue
        pos = _safe_pos(pf)
        if pos is None:
            continue
        lane = _lane_by_position_xy(pos[0], pos[1])
        counts[lane] += 1
        total += 1

    if total == 0:
        return {
            'jg_presence_top': 0.0,
            'jg_presence_mid': 0.0,
            'jg_presence_bot': 0.0,
            'jg_focus_lane': 'NEUTRAL',
        }

    ratios = {k: v / total for k, v in counts.items()}
    focus = max(ratios, key=ratios.get)

    # Si está muy repartido, neutral
    vals = sorted(ratios.values(), reverse=True)
    if len(vals) >= 2 and (vals[0] - vals[1]) < 0.10:
        focus = 'NEUTRAL'

    return {
        'jg_presence_top': ratios['TOP'],
        'jg_presence_mid': ratios['MID'],
        'jg_presence_bot': ratios['BOT'],
        'jg_focus_lane': focus,
    }


# -------------------------
# EVENTS: OBJETIVOS, FIGHTS, TORRES/PLATES
# -------------------------

def build_participant_to_team(info_participants):
    return {p['participantId']: p['teamId'] for p in info_participants}


def iter_timeline_events(timeline):
    """Yield (event, time_seconds)."""
    for frame in timeline['info']['frames']:
        for ev in frame.get('events', []):
            ts = ev.get('timestamp')
            if ts is None:
                continue
            yield ev, (ts / 1000.0)


def parse_objective_times(timeline, pid_to_team):
    """
    Objetivos con timestamp y conteos 0-14.
    Soporta formatos:
      - ELITE_MONSTER_KILL con monsterType
      - (fallback) DRAGON_KILL / RIFT_HERALD_KILL / HORDE_KILL
    """
    out = {
        100: {"first_dragon": None, "first_herald": None, "first_grubs": None,
              "drag_0_14": 0, "herald_0_14": 0, "grubs_0_14": 0},
        200: {"first_dragon": None, "first_herald": None, "first_grubs": None,
              "drag_0_14": 0, "herald_0_14": 0, "grubs_0_14": 0},
    }

    for ev, t in iter_timeline_events(timeline):
        et = ev.get("type")
        killer = ev.get("killerId", 0)
        if killer <= 0:
            continue
        team = pid_to_team.get(killer)
        if team not in (100, 200):
            continue

        # Formato nuevo (común)
        if et == "ELITE_MONSTER_KILL":
            mtype = ev.get("monsterType")  # DRAGON / RIFTHERALD / HORDE / BARON_NASHOR...
            if mtype == "DRAGON":
                if out[team]["first_dragon"] is None:
                    out[team]["first_dragon"] = t
                if t <= 14 * 60:
                    out[team]["drag_0_14"] += 1

            elif mtype == "RIFTHERALD":
                if out[team]["first_herald"] is None:
                    out[team]["first_herald"] = t
                if t <= 14 * 60:
                    out[team]["herald_0_14"] += 1

            elif mtype == "HORDE":  # Voidgrubs
                if out[team]["first_grubs"] is None:
                    out[team]["first_grubs"] = t
                if t <= 14 * 60:
                    out[team]["grubs_0_14"] += 1

            continue

        # Fallback (si Riot devuelve estos types en tu región/patch)
        if et == "DRAGON_KILL":
            if out[team]["first_dragon"] is None:
                out[team]["first_dragon"] = t
            if t <= 14 * 60:
                out[team]["drag_0_14"] += 1

        elif et == "RIFT_HERALD_KILL":
            if out[team]["first_herald"] is None:
                out[team]["first_herald"] = t
            if t <= 14 * 60:
                out[team]["herald_0_14"] += 1

        elif et == "HORDE_KILL":
            if out[team]["first_grubs"] is None:
                out[team]["first_grubs"] = t
            if t <= 14 * 60:
                out[team]["grubs_0_14"] += 1

    return out


def parse_tower_and_plate_events(timeline, pid_to_team):
    """Torres y plates por lane y ventana temporal."""
    windows = {
        '0_14': (0, 14 * 60),
        '14_25': (14 * 60, 25 * 60),
    }

    def init_bucket():
        return {
            'towers': {'TOP': 0, 'MID': 0, 'BOT': 0},
            'plates': {'TOP': 0, 'MID': 0, 'BOT': 0},
            'first_tower_time': None,
        }

    out = {
        100: {w: init_bucket() for w in windows},
        200: {w: init_bucket() for w in windows},
    }

    for ev, t in iter_timeline_events(timeline):
        et = ev.get('type')

        # ---- Torre
        if et == 'BUILDING_KILL':
            if ev.get('buildingType') != 'TOWER_BUILDING':
                continue
            killer = ev.get('killerId', 0)
            if killer <= 0:
                continue
            team = pid_to_team.get(killer)
            if team not in (100, 200):
                continue
            pos = _safe_pos(ev)
            if pos is None:
                continue
            lane = _lane_by_position_xy(pos[0], pos[1])

            # primer tower (feature auxiliar)
            ft = out[team]['0_14']['first_tower_time']
            if ft is None or t < ft:
                out[team]['0_14']['first_tower_time'] = t

            for w, (a, b) in windows.items():
                if a <= t <= b:
                    out[team][w]['towers'][lane] += 1

        # ---- Plates
        elif et == 'TURRET_PLATE_DESTROYED':
            killer = ev.get('killerId', 0)
            if killer <= 0:
                continue
            team = pid_to_team.get(killer)
            if team not in (100, 200):
                continue
            pos = _safe_pos(ev)
            if pos is None:
                continue
            lane = _lane_by_position_xy(pos[0], pos[1])
            if 0 <= t <= 14 * 60:
                out[team]['0_14']['plates'][lane] += 1

    # Propagamos first_tower_time como feature accesible
    for team in (100, 200):
        ft = out[team]['0_14']['first_tower_time']
        out[team]['14_25']['first_tower_time'] = ft

    return out


def parse_kills_and_fights(timeline, pid_to_team, fight_gap_s=25.0, end_min=25):
    """
    Agrupa CHAMPION_KILL en fights por ventana temporal (fight_gap_s).
    Clasifica cada fight como PICK / SKIRMISH / TEAMFIGHT usando:
      - num_kills en el fight
      - unique_participants agregados (killer+assists+victims de todas las kills)
      - duración del fight (end-start)

    Devuelve métricas por equipo para:
      - 0_14 y 14_25
      - además un resumen 0_25
    """
    end_t = end_min * 60

    # ---- recolectar kill events
    # guardamos también participants set por kill para poder unirlos por fight
    kill_events = []  # (t, team_of_killer, participants_set)
    kills_0_14 = {100: 0, 200: 0}
    kills_14_25 = {100: 0, 200: 0}

    for ev, t in iter_timeline_events(timeline):
        if t > end_t:
            continue
        if ev.get("type") != "CHAMPION_KILL":
            continue

        killer = ev.get("killerId", 0)
        if killer <= 0:
            continue
        team = pid_to_team.get(killer)
        if team not in (100, 200):
            continue

        # kills por ventana temporal para features
        if t <= 14 * 60:
            kills_0_14[team] += 1
        elif 14 * 60 < t <= 25 * 60:
            kills_14_25[team] += 1

        participants = set()
        participants.add(killer)
        victim = ev.get("victimId", 0)
        if victim > 0:
            participants.add(victim)
        for a in (ev.get("assistingParticipantIds") or []):
            participants.add(a)

        kill_events.append((t, team, participants))

    kill_events.sort(key=lambda x: x[0])

    # ---- agrupar kills en fights por gap temporal
    fights = []
    cur = None
    for t, team, pset in kill_events:
        if cur is None:
            cur = {
                "start": t,
                "end": t,
                "teams": set([team]),
                "kills": 1,
                "participants": set(pset),
            }
            continue

        if (t - cur["end"]) <= fight_gap_s:
            cur["end"] = t
            cur["teams"].add(team)
            cur["kills"] += 1
            cur["participants"].update(pset)
        else:
            fights.append(cur)
            cur = {
                "start": t,
                "end": t,
                "teams": set([team]),
                "kills": 1,
                "participants": set(pset),
            }
    if cur is not None:
        fights.append(cur)

    # ---- clasificación de fights
    def classify_fight(f):
        """
        Regla robusta:
          TEAMFIGHT si:
            - kills >= 3  (muy típico de TF reales)
            OR
            - unique_participants >= 6
          SKIRMISH si:
            - kills == 2 y dura poco (<=10s)  o  unique_participants == 5
          PICK si:
            - kills == 1 y unique_participants pequeño
        """
        num_kills = f["kills"]
        uniq = len(f["participants"])
        dur = f["end"] - f["start"]

        if num_kills >= 3 or uniq >= 6:
            return "TEAMFIGHT"
        if (num_kills == 2 and dur <= 10) or uniq == 5:
            return "SKIRMISH"
        return "PICK"

    for f in fights:
        f["type"] = classify_fight(f)
        f["uniq_participants"] = len(f["participants"])
        f["duration"] = f["end"] - f["start"]

    # ---- agregación por ventanas
    windows = {
        "0_14": (0, 14 * 60),
        "14_25": (14 * 60, 25 * 60),
        "0_25": (0, 25 * 60),
    }

    def init_team_bucket():
        return {
            "total_fights": 0,
            "pick_fights": 0,
            "skirmish_fights": 0,
            "teamfight_fights": 0,
            "total_kills_in_fights": 0,
            "sum_uniq_participants": 0,
            "sum_duration": 0.0,
            "first_fight_time": None,
            "first_fight_type": None,
        }

    out = {w: {100: init_team_bucket(), 200: init_team_bucket()} for w in windows}

    for f in fights:
        for w, (a, b) in windows.items():
            if not (a <= f["start"] <= b):
                continue

            for team in f["teams"]:
                tb = out[w][team]
                tb["total_fights"] += 1
                tb["total_kills_in_fights"] += f["kills"]
                tb["sum_uniq_participants"] += f["uniq_participants"]
                tb["sum_duration"] += f["duration"]

                if tb["first_fight_time"] is None or f["start"] < tb["first_fight_time"]:
                    tb["first_fight_time"] = f["start"]
                    tb["first_fight_type"] = f["type"]

                if f["type"] == "PICK":
                    tb["pick_fights"] += 1
                elif f["type"] == "SKIRMISH":
                    tb["skirmish_fights"] += 1
                elif f["type"] == "TEAMFIGHT":
                    tb["teamfight_fights"] += 1

    # ---- convertir a features finales (ratios, promedios)
    def finalize(tb, window_minutes):
        total = tb["total_fights"]
        if total == 0:
            return {
                "total_fights": 0,
                "pick_fights": 0,
                "skirmish_fights": 0,
                "teamfight_fights": 0,
                "pick_ratio": 0.0,
                "skirmish_ratio": 0.0,
                "teamfight_ratio": 0.0,
                "fights_per_min": 0.0,
                "avg_kills_per_fight": 0.0,
                "avg_uniq_participants": 0.0,
                "avg_fight_duration": 0.0,
                "first_fight_time": tb["first_fight_time"],
                "first_fight_type": tb["first_fight_type"],
            }

        return {
            "total_fights": total,
            "pick_fights": tb["pick_fights"],
            "skirmish_fights": tb["skirmish_fights"],
            "teamfight_fights": tb["teamfight_fights"],
            "pick_ratio": tb["pick_fights"] / total,
            "skirmish_ratio": tb["skirmish_fights"] / total,
            "teamfight_ratio": tb["teamfight_fights"] / total,
            "fights_per_min": total / max(window_minutes, 1e-6),
            "avg_kills_per_fight": tb["total_kills_in_fights"] / total,
            "avg_uniq_participants": tb["sum_uniq_participants"] / total,
            "avg_fight_duration": tb["sum_duration"] / total,
            "first_fight_time": tb["first_fight_time"],
            "first_fight_type": tb["first_fight_type"],
        }

    # minutos por ventana
    win_minutes = {"0_14": 14, "14_25": 11, "0_25": 25}

    final = {}
    for w in windows.keys():
        final[w] = {}
        for team in (100, 200):
            final[w][team] = finalize(out[w][team], win_minutes[w])

    # empaquetamos todo en un dict plano para comodidad
    return {
        "kills_0_14": kills_0_14,
        "kills_14_25": kills_14_25,
        "fights": final,
    }


def derive_team_labels(team_id, obj, towers, fights, cluster):
    """Labels (targets) derivados por equipo."""
    # L1 FirstMajorObjective_0_14
    candidates = []
    if obj[team_id]['first_dragon'] is not None and obj[team_id]['first_dragon'] <= 14 * 60:
        candidates.append(('DRAGON', obj[team_id]['first_dragon']))
    if obj[team_id]['first_grubs'] is not None and obj[team_id]['first_grubs'] <= 14 * 60:
        candidates.append(('GRUBS', obj[team_id]['first_grubs']))
    if obj[team_id]['first_herald'] is not None and obj[team_id]['first_herald'] <= 14 * 60:
        candidates.append(('HERALD', obj[team_id]['first_herald']))

    first_major = 'NONE' if not candidates else min(candidates, key=lambda x: x[1])[0]

    # L2 ObjectivePriority_0_14
    n_drag = obj[team_id]['drag_0_14']
    n_grub = obj[team_id]['grubs_0_14']
    n_her = obj[team_id]['herald_0_14']
    total_obj = n_drag + n_grub + n_her

    if total_obj == 0:
        obj_prio = 'NONE'
    elif n_drag >= (n_grub + n_her) + 1:
        obj_prio = 'DRAGONS'
    elif (n_grub + n_her) >= n_drag + 1:
        obj_prio = 'GRUBS_HERALD'
    else:
        obj_prio = 'MIXED'

    # L3 SkirmishStyle_0_25
    f025 = fights["fights"]["0_25"][team_id]
    total_f = f025["total_fights"]
    pick_f = f025["pick_fights"]
    tf_f = f025["teamfight_fights"]
    sk_f = f025["skirmish_fights"]

    if total_f == 0:
        sk_style = "LOW_ACTION"
    elif tf_f >= pick_f + 2 and tf_f >= sk_f:
        sk_style = "TEAMFIGHT"
    elif pick_f >= tf_f + 2 and pick_f >= sk_f:
        sk_style = "PICK"
    else:
        sk_style = "MIXED"

    # L4 FirstFightType_0_14
    ff_time = fights["fights"]["0_14"][team_id]["first_fight_time"]
    ff_type = fights["fights"]["0_14"][team_id]["first_fight_type"]
    if ff_time is None or ff_time > 14 * 60 or ff_type is None:
        first_fight = "NONE"
    else:
        # ff_type será PICK / SKIRMISH / TEAMFIGHT
        # si quieres solo PICK vs TEAMFIGHT, mapea SKIRMISH a NONE o PICK
        if ff_type == "TEAMFIGHT":
            first_fight = "TEAMFIGHT"
        elif ff_type == "PICK":
            first_fight = "PICK"
        else:
            first_fight = "NONE"

    # L7 Tempo_0_20
    first_tower = towers[team_id]['0_14']['first_tower_time']
    kills_0_14 = fights['kills_0_14'][team_id]
    obj_0_14 = total_obj

    score = 0
    if first_tower is not None and first_tower <= 12 * 60:
        score += 1
    if obj_0_14 >= 2:
        score += 1
    if kills_0_14 >= 8:
        score += 1

    if score >= 2:
        tempo = 'FAST'
    elif score == 1:
        tempo = 'MID'
    else:
        tempo = 'SLOW'

    # L11 Grouping_15
    gr15 = cluster.get('Team_GroupedRatio_15')
    if gr15 is None:
        grouping = 'MIXED'
    elif gr15 >= 0.70:
        grouping = 'GROUP'
    elif gr15 <= 0.55:
        grouping = 'SPLIT'
    else:
        grouping = 'MIXED'

    # L5 TowerPlan_14_25
    t14_25 = towers[team_id]['14_25']['towers']
    top_bot = t14_25['TOP'] + t14_25['BOT']
    mid = t14_25['MID']

    if gr15 is None:
        tower_plan = 'UNKNOWN'
    elif gr15 <= 0.55 and top_bot >= mid + 1:
        tower_plan = 'SPLIT_1_3_1'
    elif gr15 >= 0.70 and mid >= top_bot:
        tower_plan = 'SIEGE_GROUP'
    else:
        tower_plan = 'STANDARD'

    # L6 LanePressureBias_0_14 (plates)
    p0_14 = towers[team_id]['0_14']['plates']
    if (p0_14['TOP'] + p0_14['MID'] + p0_14['BOT']) == 0:
        lane_bias = 'NEUTRAL'
    else:
        vals = [('TOP', p0_14['TOP']), ('MID', p0_14['MID']), ('BOT', p0_14['BOT'])]
        vals.sort(key=lambda x: x[1], reverse=True)
        lane_bias = vals[0][0] if vals[0][1] >= vals[1][1] + 2 else 'NEUTRAL'

    # L8 Snowball_0_14
    snowball = 'YES' if ((first_tower is not None and first_tower <= 14 * 60) or obj_0_14 >= 2) else 'NO'

    return {
        'Label_FirstMajorObjective_0_14': first_major,
        'Label_ObjectivePriority_0_14': obj_prio,
        'Label_SkirmishStyle_0_25': sk_style,
        'Label_FirstFightType_0_14': first_fight,
        'Label_TowerPlan_14_25': tower_plan,
        'Label_LanePressureBias_0_14': lane_bias,
        'Label_Tempo_0_20': tempo,
        'Label_Grouping_15': grouping,
        'Label_Snowball_0_14': snowball,
    }


# -------------------------
# SNAPSHOTS TEMPORALES (XP, GOLD, CS, TAKEDOWNS)
# -------------------------

def get_timeline_stats_by_minute(timeline, minutes_to_track=[5, 10, 15, 20]):
    """
    Procesa la timeline para obtener snapshots de XP, Gold, CS y Takedowns
    en los minutos indicados.
    """
    frames = timeline['info']['frames']
    snapshots = {}

    # Contadores de Kills/Assists (Takedowns) acumulativos
    takedowns_counter = {pid: 0 for pid in range(1, 11)}

    for i, frame in enumerate(frames):
        # A. Procesar Eventos para contar Takedowns
        for event in frame['events']:
            if event['type'] == 'CHAMPION_KILL':
                killer_id = event.get('killerId', 0)
                if killer_id > 0:
                    takedowns_counter[killer_id] += 1

                if 'assistingParticipantIds' in event:
                    for assist_id in event['assistingParticipantIds']:
                        takedowns_counter[assist_id] += 1

        # B. Guardar Snapshot si coincide con el minuto
        if i in minutes_to_track:
            snapshots[i] = {}
            for pid_str, p_data in frame['participantFrames'].items():
                pid = int(pid_str)
                snapshots[i][pid] = {
                    'gold': p_data['totalGold'],
                    'xp': p_data['xp'],
                    'cs': p_data['minionsKilled'] + p_data['jungleMinionsKilled'],
                    'jungle_cs': p_data['jungleMinionsKilled'],
                    'lane_cs': p_data['minionsKilled'],
                    'takedowns': takedowns_counter[pid],
                }

    # Rellenar minutos faltantes con el último estado conocido
    last_frame_idx = len(frames) - 1
    last_known_data = {}

    for pid_str, p_data in frames[last_frame_idx]['participantFrames'].items():
        pid = int(pid_str)
        last_known_data[pid] = {
            'gold': p_data['totalGold'],
            'xp': p_data['xp'],
            'cs': p_data['minionsKilled'] + p_data['jungleMinionsKilled'],
            'jungle_cs': p_data['jungleMinionsKilled'],
            'lane_cs': p_data['minionsKilled'],
            'takedowns': takedowns_counter[pid],
        }

    for m in minutes_to_track:
        if m not in snapshots:
            snapshots[m] = last_known_data

    return snapshots


def encode_summoners_unordered(s1, s2):
    # IDs estándar (pueden variar, pero estos suelen ser estables)
    SPELLS = {
        1: "CLEANSE",
        3: "EXHAUST",
        4: "FLASH",
        6: "GHOST",
        7: "HEAL",
        11: "SMITE",
        12: "TELEPORT",
        13: "CLARITY",
        14: "IGNITE",
        21: "BARRIER",
        32: "SNOWBALL",
    }
    present = set([s1, s2])
    out = {}
    for sid, name in SPELLS.items():
        out[f"SS_{name}"] = 1 if sid in present else 0
    # combo legible (ordenado)
    names = sorted([SPELLS.get(s1, f"ID{s1}"), SPELLS.get(s2, f"ID{s2}")])
    out["SS_Combo"] = "+".join(names)
    return out


def process_match(match_id):
    try:
        details = watcher.match.by_id(REGION, match_id)
        timeline = watcher.match.timeline_by_match(REGION, match_id)
    except ApiError as err:
        if err.response.status_code == 404:
            print(f"Partida {match_id} no encontrada.")
        else:
            print(f"Error bajando {match_id}: {err}")
        return []

    # --- 1. DATOS GENERALES ---
    info = details['info']
    duration = info['gameDuration'] / 60  # minutos

    if duration < 15 or info.get('gameMode') != 'CLASSIC':
        return []

    # Identificar equipos
    teams_info = {t['teamId']: t for t in info['teams']}
    winning_team_id = 100 if teams_info[100]['win'] else 200

    # Calcular Kills Totales por Equipo (Para Kill Participation)
    team_kills = {100: 0, 200: 0}
    for p in info['participants']:
        team_kills[p['teamId']] += p['kills']

    # --- 2. SNAPSHOTS TEMPORALES ---
    time_minutes = [5, 10, 15, 20]
    snapshots = get_timeline_stats_by_minute(timeline, time_minutes)

    # --- 2B. POSICIONES + EVENTS (por partida) ---
    pos_minutes = [5, 10, 15, 20]
    positions = get_positions_by_minute(timeline, pos_minutes)

    pid_to_team = build_participant_to_team(info['participants'])
    obj = parse_objective_times(timeline, pid_to_team)
    towers = parse_tower_and_plate_events(timeline, pid_to_team)
    fights = parse_kills_and_fights(timeline, pid_to_team, fight_gap_s=25.0, end_min=25)

    # Mapear participantes
    pmap = {p['participantId']: p for p in info['participants']}
    rows_generated = []

    for focus_team_id in [100, 200]:
        did_win = 1 if focus_team_id == winning_team_id else 0

        row = {
            'matchId': match_id,
            'teamId': focus_team_id,
            'win': did_win,
            'gameDuration': duration,
            'gameVersion': info.get('gameVersion', None),
            'queueId': info.get('queueId', None),
        }

        roles_order = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
        valid_team = True

        # Iterar por roles
        for role in roles_order:
            ally_p = None
            enemy_p = None

            for pid, p in pmap.items():
                if p.get('teamPosition') == role:
                    if p['teamId'] == focus_team_id:
                        ally_p = p
                    else:
                        enemy_p = p

            if not ally_p or not enemy_p:
                valid_team = False
                break

            prefix = role
            pid_ally = ally_p['participantId']
            pid_enemy = enemy_p['participantId']

            # A. IDENTIFICADORES
            row[f"{prefix}_Ally_ID"] = ally_p['championId']
            row[f"{prefix}_Enemy_ID"] = enemy_p['championId']

            # (Opcional) Summoners / runas para futuro (pre-game)
            s1 = ally_p.get("summoner1Id", None)
            s2 = ally_p.get("summoner2Id", None)
            row[f"{prefix}_Summoner1"] = s1
            row[f"{prefix}_Summoner2"] = s2

            if s1 is not None and s2 is not None:
                enc = encode_summoners_unordered(s1, s2)
                for k, v in enc.items():
                    row[f"{prefix}_{k}"] = v

            perks = ally_p.get('perks', {})
            styles = perks.get('styles', []) if isinstance(perks, dict) else []
            row[f"{prefix}_Keystone"] = None
            if styles:
                # El primer style suele contener la keystone
                try:
                    selections = styles[0].get('selections', [])
                    if selections:
                        row[f"{prefix}_Keystone"] = selections[0].get('perk', None)
                except Exception:
                    pass

            # B. SNAPSHOTS (Diffs @ 5, 10, 15, 20)
            for m in time_minutes:
                s_ally = snapshots[m][pid_ally]
                s_enemy = snapshots[m][pid_enemy]

                row[f"{prefix}_GoldDiff_{m}"] = s_ally['gold'] - s_enemy['gold']
                row[f"{prefix}_XpDiff_{m}"] = s_ally['xp'] - s_enemy['xp']
                row[f"{prefix}_CsDiff_{m}"] = s_ally['cs'] - s_enemy['cs']

            row[f"{prefix}_EarlyTakedowns"] = snapshots[15][pid_ally]['takedowns']

            # C. ESTADÍSTICAS ROL (Early)
            if role == 'JUNGLE':
                row[f"{prefix}_JgCsBefore10"] = snapshots[10][pid_ally]['jungle_cs']
                row[f"{prefix}_EnemyJgInvades"] = ally_p.get('challenges', {}).get('enemyJungleMonsterKills', 0)
            else:
                row[f"{prefix}_LaneCsBefore10"] = snapshots[10][pid_ally]['lane_cs']

            # D. ESTADÍSTICAS GLOBALES (post-game; útiles para analítica)
            stats = ally_p
            challenges = ally_p.get('challenges', {})

            # Daño (por minuto)
            row[f"{prefix}_DmgTotal"] = stats['totalDamageDealtToChampions'] / duration
            row[f"{prefix}_DmgPhys"] = stats['physicalDamageDealtToChampions'] / duration
            row[f"{prefix}_DmgMagic"] = stats['magicDamageDealtToChampions'] / duration
            row[f"{prefix}_DmgTrue"] = stats['trueDamageDealtToChampions'] / duration
            row[f"{prefix}_DmgTurret"] = stats['damageDealtToTurrets'] / duration
            row[f"{prefix}_DmgObj"] = stats['damageDealtToObjectives'] / duration

            # Economía
            total_cs = stats['totalMinionsKilled'] + stats['neutralMinionsKilled']
            row[f"{prefix}_TotalCS"] = total_cs
            row[f"{prefix}_GoldEarned"] = stats['goldEarned']

            # Objective Participation
            row[f"{prefix}_DragonTakedowns"] = challenges.get('dragonTakedowns', 0)
            row[f"{prefix}_BaronTakedowns"] = challenges.get('baronTakedowns', 0)
            row[f"{prefix}_RiftHeraldTakedowns"] = challenges.get('riftHeraldTakedowns', 0)
            row[f"{prefix}_VoidMonsterTakedowns"] = challenges.get('voidMonsterKill', 0)

            # Desafíos Varios
            row[f"{prefix}_TurretPlates"] = challenges.get('turretPlatesTaken', 0)
            row[f"{prefix}_MaxCsAdvantage"] = challenges.get('maxCsAdvantageOnLaneOpponent', 0)
            row[f"{prefix}_SoloKills"] = challenges.get('soloKills', 0)

            # Visión (por minuto)
            row[f"{prefix}_VisionScore"] = stats['visionScore'] / duration
            row[f"{prefix}_WardsPlaced"] = stats['wardsPlaced'] / duration
            row[f"{prefix}_WardsKilled"] = stats['wardsKilled'] / duration
            row[f"{prefix}_ControlWardsPlaced"] = challenges.get('controlWardsPlaced', 0) / duration

            # CC
            row[f"{prefix}_TimeCC"] = stats['timeCCingOthers'] / duration
            row[f"{prefix}_TotalCC"] = stats['totalTimeCCDealt'] / duration

            # Curaciones y escudos
            row[f"{prefix}_TotalHeal"] = stats['totalHeal'] / duration
            row[f"{prefix}_HealOnTeammates"] = stats['totalHealsOnTeammates'] / duration
            row[f"{prefix}_DamageShielded"] = stats['totalDamageShieldedOnTeammates'] / duration

            # Mitigación y daño recibido
            row[f"{prefix}_DamageTaken"] = stats['totalDamageTaken'] / duration
            row[f"{prefix}_DamageMitigated"] = stats['damageSelfMitigated'] / duration

            # KP
            my_kills_assists = stats['kills'] + stats['assists']
            total_team_kills = team_kills[focus_team_id]
            kp = (my_kills_assists / total_team_kills) if total_team_kills > 0 else 0
            row[f"{prefix}_KillParticipation"] = round(kp, 2)

        if not valid_team:
            return []

        # -------------------------
        # FEATURES DE POSICIÓN / CLUSTER (por equipo)
        # -------------------------
        for m in pos_minutes:
            cm = compute_team_cluster_metrics(
                positions_at_minute=positions[m],
                participants_info=info['participants'],
                team_id=focus_team_id,
                grouped_radius=2200.0,
            )
            row[f"Team_Spread_{m}"] = cm['spread_mean']
            row[f"Team_GroupedRatio_{m}"] = cm['grouped_ratio']

        # Jungle focus lane (label + features)
        jg = compute_jungle_lane_presence(
            timeline=timeline,
            participants_info=info['participants'],
            team_id=focus_team_id,
            start_min=2,
            end_min=14,
        )
        row['Feat_JUNGLE_PresenceTop_2_14'] = jg['jg_presence_top']
        row['Feat_JUNGLE_PresenceMid_2_14'] = jg['jg_presence_mid']
        row['Feat_JUNGLE_PresenceBot_2_14'] = jg['jg_presence_bot']
        row['Label_JungleFocusLane_2_14'] = jg['jg_focus_lane']

        # -------------------------
        # FEATURES AUXILIARES: OBJETIVOS / TORRES / FIGHTS
        # -------------------------
        row['Feat_FirstDragonTime'] = obj[focus_team_id]['first_dragon']
        row['Feat_FirstGrubsTime'] = obj[focus_team_id]['first_grubs']
        row['Feat_FirstHeraldTime'] = obj[focus_team_id]['first_herald']
        row['Feat_FirstTowerTime'] = towers[focus_team_id]['0_14']['first_tower_time']

        row['Feat_Dragons_0_14'] = obj[focus_team_id]['drag_0_14']
        row['Feat_Grubs_0_14'] = obj[focus_team_id]['grubs_0_14']
        row['Feat_Heralds_0_14'] = obj[focus_team_id]['herald_0_14']
        row['Feat_Objectives_0_14'] = row['Feat_Dragons_0_14'] + row['Feat_Grubs_0_14'] + row['Feat_Heralds_0_14']

        # Kills por ventanas
        row["Feat_Kills_0_14"] = fights["kills_0_14"][focus_team_id]
        row["Feat_Kills_14_25"] = fights["kills_14_25"][focus_team_id]

        # Features fights 0-14
        f014 = fights["fights"]["0_14"][focus_team_id]
        row["Feat_Fights_0_14"] = f014["total_fights"]
        row["Feat_PickFights_0_14"] = f014["pick_fights"]
        row["Feat_SkirmishFights_0_14"] = f014["skirmish_fights"]
        row["Feat_TeamfightFights_0_14"] = f014["teamfight_fights"]
        row["Feat_PickRatio_0_14"] = f014["pick_ratio"]
        row["Feat_TeamfightRatio_0_14"] = f014["teamfight_ratio"]
        row["Feat_FightsPerMin_0_14"] = f014["fights_per_min"]
        row["Feat_AvgKillsPerFight_0_14"] = f014["avg_kills_per_fight"]
        row["Feat_AvgUniqParticipants_0_14"] = f014["avg_uniq_participants"]
        row["Feat_AvgFightDuration_0_14"] = f014["avg_fight_duration"]

        # Features fights 14-25
        f1425 = fights["fights"]["14_25"][focus_team_id]
        row["Feat_Fights_14_25"] = f1425["total_fights"]
        row["Feat_PickFights_14_25"] = f1425["pick_fights"]
        row["Feat_SkirmishFights_14_25"] = f1425["skirmish_fights"]
        row["Feat_TeamfightFights_14_25"] = f1425["teamfight_fights"]
        row["Feat_PickRatio_14_25"] = f1425["pick_ratio"]
        row["Feat_TeamfightRatio_14_25"] = f1425["teamfight_ratio"]
        row["Feat_FightsPerMin_14_25"] = f1425["fights_per_min"]
        row["Feat_AvgKillsPerFight_14_25"] = f1425["avg_kills_per_fight"]
        row["Feat_AvgUniqParticipants_14_25"] = f1425["avg_uniq_participants"]
        row["Feat_AvgFightDuration_14_25"] = f1425["avg_fight_duration"]

        # Features fights 0-25 (resumen)
        f025 = fights["fights"]["0_25"][focus_team_id]
        row["Feat_Fights_0_25"] = f025["total_fights"]
        row["Feat_PickRatio_0_25"] = f025["pick_ratio"]
        row["Feat_TeamfightRatio_0_25"] = f025["teamfight_ratio"]
        row["Feat_FightsPerMin_0_25"] = f025["fights_per_min"]
        row["Feat_AvgKillsPerFight_0_25"] = f025["avg_kills_per_fight"]
        row["Feat_AvgUniqParticipants_0_25"] = f025["avg_uniq_participants"]

        # Primer fight EARLY (0-14)
        row["Feat_FirstFightTime_0_14"] = f014["first_fight_time"]
        row["Feat_FirstFightType_0_14"] = f014["first_fight_type"]


        for lane in ('TOP', 'MID', 'BOT'):
            row[f"Feat_TowerKills_{lane}_0_14"] = towers[focus_team_id]['0_14']['towers'][lane]
            row[f"Feat_TowerKills_{lane}_14_25"] = towers[focus_team_id]['14_25']['towers'][lane]
            row[f"Feat_Plates_{lane}_0_14"] = towers[focus_team_id]['0_14']['plates'][lane]

        # -------------------------
        # LABELS (targets) DERIVADOS
        # -------------------------
        cluster_ctx = {'Team_GroupedRatio_15': row.get('Team_GroupedRatio_15')}
        labels = derive_team_labels(
            team_id=focus_team_id,
            obj=obj,
            towers=towers,
            fights=fights,
            cluster=cluster_ctx,
        )
        row.update(labels)

        # --- TARGETS EQUIPO (del match info, finales) ---
        t_stats = next(t for t in info['teams'] if t['teamId'] == focus_team_id)
        objs = t_stats['objectives']

        row['Team_Dragons'] = objs.get('dragon', {}).get('kills', 0)
        row['Team_Barons'] = objs.get('baron', {}).get('kills', 0)
        row['Team_Towers'] = objs.get('tower', {}).get('kills', 0)
        row['Team_Inhibitors'] = objs.get('inhibitor', {}).get('kills', 0)
        row['Team_RiftHeralds'] = objs.get('riftHerald', {}).get('kills', 0)
        row['Team_VoidGrubs'] = objs.get('horde', {}).get('kills', 0)

        winner_sample = next((p for p in info['participants'] if p['teamId'] == focus_team_id), None)
        if winner_sample:
            row['Team_ElderDragons'] = winner_sample.get('challenges', {}).get('teamElderDragonKills', 0)
        else:
            row['Team_ElderDragons'] = 0

        rows_generated.append(row)

    return rows_generated


if __name__ == '__main__':
    target_puuid = account_watcher.account.by_riot_id(REGION, GAME_NAME, TAG_LINE).get('puuid')
    if target_puuid:
        print(f"Iniciando prueba de extracción para {GAME_NAME}#{TAG_LINE}...")
        match_ids = get_match_ids(target_puuid, count=10, ranked_only=True)
        dataset = []
        for i, m in enumerate(match_ids):
            print(f"[{i + 1}/{len(match_ids)}] ", end='')

            rows = process_match(m)
            if rows:
                dataset.extend(rows)
            time.sleep(1.3)

        if dataset:
            pd.DataFrame(dataset).to_csv('Data/dataset_test_v2.csv', index=False)
            print('Prueba finalizada con éxito.')
    else:
        print(
            f"ERROR: No se pudo encontrar el PUUID para {GAME_NAME}#{TAG_LINE}. Verifica el nombre, región y la clave API."
        )
