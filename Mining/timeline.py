from .position import _safe_pos, _lane_by_position_xy


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
    """Objetivos con timestamp y conteos 0-14 (soporta ELITE_MONSTER_KILL)."""
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

        if et == "ELITE_MONSTER_KILL":
            mtype = ev.get("monsterType")
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
            elif mtype == "HORDE":
                if out[team]["first_grubs"] is None:
                    out[team]["first_grubs"] = t
                if t <= 14 * 60:
                    out[team]["grubs_0_14"] += 1
            continue

        # fallback por si te aparece en tu región
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
    windows = {'0_14': (0, 14 * 60), '14_25': (14 * 60, 25 * 60)}

    def init_bucket():
        return {
            'towers': {'TOP': 0, 'MID': 0, 'BOT': 0},
            'plates': {'TOP': 0, 'MID': 0, 'BOT': 0},
            'first_tower_time': None,
        }

    out = {100: {w: init_bucket() for w in windows}, 200: {w: init_bucket() for w in windows}}

    for ev, t in iter_timeline_events(timeline):
        et = ev.get('type')

        if et == 'BUILDING_KILL':
            if ev.get('buildingType') != 'TOWER_BUILDING':
                continue
            killer = ev.get('killerId', 0)
            if killer <= 0:
                continue
            team = pid_to_team.get(killer)
            if team not in (100, 200):
                continue

            if ev.get("laneType") in ("TOP_LANE", "MID_LANE", "BOT_LANE"):
                lane = {"TOP_LANE": "TOP", "MID_LANE": "MID", "BOT_LANE": "BOT"}[ev["laneType"]]
            else:
                pos = _safe_pos(ev)
                if pos is None:
                    continue
                lane = _lane_by_position_xy(pos[0], pos[1])

            ft = out[team]['0_14']['first_tower_time']
            if ft is None or t < ft:
                out[team]['0_14']['first_tower_time'] = t

            for w, (a, b) in windows.items():
                if a <= t <= b:
                    out[team][w]['towers'][lane] += 1

        elif et == 'TURRET_PLATE_DESTROYED':
            killer = ev.get('killerId', 0)
            if killer <= 0:
                continue
            team = pid_to_team.get(killer)
            if team not in (100, 200):
                continue

            if ev.get("laneType") in ("TOP_LANE", "MID_LANE", "BOT_LANE"):
                lane = {"TOP_LANE": "TOP", "MID_LANE": "MID", "BOT_LANE": "BOT"}[ev["laneType"]]
            else:
                pos = _safe_pos(ev)
                if pos is None:
                    continue
                lane = _lane_by_position_xy(pos[0], pos[1])

            if 0 <= t <= 14 * 60:
                out[team]['0_14']['plates'][lane] += 1

    for team in (100, 200):
        ft = out[team]['0_14']['first_tower_time']
        out[team]['14_25']['first_tower_time'] = ft

    return out


def parse_kills_and_fights(timeline, pid_to_team, fight_gap_s=25.0, end_min=25):
    """Fights robustos (0-14 / 14-25 / 0-25)."""
    end_t = end_min * 60

    kill_events = []
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

        if t <= 14 * 60:
            kills_0_14[team] += 1
        elif 14 * 60 < t <= 25 * 60:
            kills_14_25[team] += 1

        participants = {killer}
        victim = ev.get("victimId", 0)
        if victim > 0:
            participants.add(victim)
        for a in (ev.get("assistingParticipantIds") or []):
            participants.add(a)

        kill_events.append((t, team, participants))

    kill_events.sort(key=lambda x: x[0])

    fights = []
    cur = None
    for t, team, pset in kill_events:
        if cur is None:
            cur = {"start": t, "end": t, "teams": {team}, "kills": 1, "participants": set(pset)}
            continue

        if (t - cur["end"]) <= fight_gap_s:
            cur["end"] = t
            cur["teams"].add(team)
            cur["kills"] += 1
            cur["participants"].update(pset)
        else:
            fights.append(cur)
            cur = {"start": t, "end": t, "teams": {team}, "kills": 1, "participants": set(pset)}
    if cur is not None:
        fights.append(cur)

    def classify_fight(f):
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

    windows = {"0_14": (0, 14 * 60), "14_25": (14 * 60, 25 * 60), "0_25": (0, 25 * 60)}

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
                else:
                    tb["teamfight_fights"] += 1

    def finalize(tb, window_minutes):
        total = tb["total_fights"]
        if total == 0:
            return {
                "total_fights": 0, "pick_fights": 0, "skirmish_fights": 0, "teamfight_fights": 0,
                "pick_ratio": 0.0, "skirmish_ratio": 0.0, "teamfight_ratio": 0.0,
                "fights_per_min": 0.0, "avg_kills_per_fight": 0.0, "avg_uniq_participants": 0.0,
                "avg_fight_duration": 0.0, "first_fight_time": tb["first_fight_time"],
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

    win_minutes = {"0_14": 14, "14_25": 11, "0_25": 25}
    final = {w: {team: finalize(out[w][team], win_minutes[w]) for team in (100, 200)} for w in windows}

    return {"kills_0_14": kills_0_14, "kills_14_25": kills_14_25, "fights": final}


# -------------------------
# SNAPSHOTS TEMPORALES
# -------------------------

def get_timeline_stats_by_minute(timeline, minutes_to_track=(5, 10, 15, 20)):
    """Snapshots por minuto: gold/xp/cs y takedowns acumulados."""
    frames = timeline['info']['frames']
    snapshots = {}
    takedowns_counter = {pid: 0 for pid in range(1, 11)}

    for i, frame in enumerate(frames):
        for event in frame.get('events', []):
            if event.get('type') == 'CHAMPION_KILL':
                killer_id = event.get('killerId', 0)
                if killer_id > 0:
                    takedowns_counter[killer_id] += 1
                for assist_id in (event.get('assistingParticipantIds') or []):
                    takedowns_counter[assist_id] += 1

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

    last_frame = frames[-1]['participantFrames']
    last_known = {}
    for pid_str, p_data in last_frame.items():
        pid = int(pid_str)
        last_known[pid] = {
            'gold': p_data['totalGold'],
            'xp': p_data['xp'],
            'cs': p_data['minionsKilled'] + p_data['jungleMinionsKilled'],
            'jungle_cs': p_data['jungleMinionsKilled'],
            'lane_cs': p_data['minionsKilled'],
            'takedowns': takedowns_counter[pid],
        }
    for m in minutes_to_track:
        if m not in snapshots:
            snapshots[m] = last_known

    return snapshots
