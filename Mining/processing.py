import time
import pandas as pd

from riotwatcher import ApiError

from ..Raw_Data.api import watcher, account_watcher, REGION, GAME_NAME, TAG_LINE, riot_call_with_retry, get_match_ids
from .timeline import (
    build_participant_to_team,
    parse_objective_times,
    parse_tower_and_plate_events,
    parse_kills_and_fights,
    get_timeline_stats_by_minute,
)
from .position import (
    get_positions_by_minute,
    compute_team_cluster_metrics,
    compute_player_group_side_metrics,
    compute_jungle_presence_and_focus,
    compute_support_roam_ratio,
)
from .labels import (
    encode_summoners_unordered,
    extract_keystone,
    derive_player_mvp_labels,
    derive_jungle_gank_labels,
    derive_support_roam_labels,
    derive_team_focus_label,
    derive_team_labels,
)


# -------------------------
# MATCH PROCESSING
# -------------------------

def process_match(match_id):
    try:
        details = riot_call_with_retry(watcher.match.by_id, REGION, match_id)
        timeline = riot_call_with_retry(watcher.match.timeline_by_match, REGION, match_id)
    except ApiError as err:
        if getattr(err.response, "status_code", None) == 404:
            print(f"Partida {match_id} no encontrada.")
        else:
            print(f"Error bajando {match_id}: {err}")
        return []

    info = details['info']
    duration_min = info.get('gameDuration', 0) / 60.0

    if duration_min < 15 or info.get('gameMode') != 'CLASSIC':
        return []

    teams_info = {t['teamId']: t for t in info['teams']}
    winning_team_id = 100 if teams_info.get(100, {}).get('win', False) else 200

    team_kills = {100: 0, 200: 0}
    for p in info['participants']:
        team_kills[p['teamId']] += p.get('kills', 0)

    time_minutes = [5, 10, 15, 20]
    snapshots = get_timeline_stats_by_minute(timeline, time_minutes)

    pos_minutes = [5, 10, 15, 20]
    positions = get_positions_by_minute(timeline, pos_minutes)

    pid_to_team = build_participant_to_team(info['participants'])
    obj = parse_objective_times(timeline, pid_to_team)
    towers = parse_tower_and_plate_events(timeline, pid_to_team)
    fights = parse_kills_and_fights(timeline, pid_to_team, fight_gap_s=25.0, end_min=25)

    pmap = {p['participantId']: p for p in info['participants']}
    rows_generated = []

    roles_order = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']

    for focus_team_id in (100, 200):
        did_win = 1 if focus_team_id == winning_team_id else 0

        row = {
            'matchId': match_id,
            'teamId': focus_team_id,
            'win': did_win,
            'meta_gameDurationMin': duration_min,
            'meta_gameVersion': info.get('gameVersion', None),
            'meta_queueId': info.get('queueId', None),
        }

        valid_team = True

        role_to_pid = {}
        for role in roles_order:
            ally_p = None
            enemy_p = None
            for p in pmap.values():
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

            role_to_pid[role] = pid_ally
            # ---- PRE-GAME INPUTS (X_)
            row[f"X_{prefix}_AllyChampId"] = ally_p.get('championId')
            row[f"X_{prefix}_EnemyChampId"] = enemy_p.get('championId')

            s1 = ally_p.get('summoner1Id', None)
            s2 = ally_p.get('summoner2Id', None)
            if s1 is not None and s2 is not None:
                row[f"X_{prefix}_SummonerA"] = min(s1, s2)
                row[f"X_{prefix}_SummonerB"] = max(s1, s2)
                for k, v in encode_summoners_unordered(s1, s2).items():
                    row[f"X_{prefix}_{k}"] = v
            else:
                row[f"X_{prefix}_SummonerA"] = None
                row[f"X_{prefix}_SummonerB"] = None

            row[f"X_{prefix}_Keystone"] = extract_keystone(ally_p)

            # ---- IN-GAME (Post_)
            for m in time_minutes:
                s_ally = snapshots[m][pid_ally]
                s_enemy = snapshots[m][pid_enemy]
                row[f"Post_{prefix}_GoldDiff_{m}"] = s_ally['gold'] - s_enemy['gold']
                row[f"Post_{prefix}_XpDiff_{m}"] = s_ally['xp'] - s_enemy['xp']
                row[f"Post_{prefix}_CsDiff_{m}"] = s_ally['cs'] - s_enemy['cs']

            row[f"Post_{prefix}_EarlyTakedowns_15"] = snapshots[15][pid_ally]['takedowns']

            if role == 'JUNGLE':
                row[f"Post_{prefix}_JgCsBefore10"] = snapshots[10][pid_ally]['jungle_cs']
                row[f"Post_{prefix}_EnemyJgInvades"] = ally_p.get('challenges', {}).get('enemyJungleMonsterKills', 0)
            else:
                row[f"Post_{prefix}_LaneCsBefore10"] = snapshots[10][pid_ally]['lane_cs']

            # ---- Post-game analytics (Post_) - opcional pero prefijado
            stats = ally_p
            row[f"Post_{prefix}_DmgTotalPM"] = stats.get('totalDamageDealtToChampions', 0) / max(duration_min, 1e-6)
            row[f"Post_{prefix}_DmgTurretPM"] = stats.get('damageDealtToTurrets', 0) / max(duration_min, 1e-6)
            row[f"Post_{prefix}_DmgObjPM"] = stats.get('damageDealtToObjectives', 0) / max(duration_min, 1e-6)
            row[f"Post_{prefix}_VisionScorePM"] = stats.get('visionScore', 0) / max(duration_min, 1e-6)

            my_kills_assists = stats.get('kills', 0) + stats.get('assists', 0)
            total_team_kills = team_kills[focus_team_id]
            kp = (my_kills_assists / total_team_kills) if total_team_kills > 0 else 0.0
            row[f"Post_{prefix}_KillParticipation"] = round(kp, 3)

        if not valid_team:
            return []

        # ---- TEAM CLUSTER FEATURES (Feat_)
        for m in pos_minutes:
            cm = compute_team_cluster_metrics(positions[m], info['participants'], focus_team_id, grouped_radius=2200.0)
            row[f"Feat_Team_Spread_{m}"] = cm['spread_mean']
            row[f"Feat_Team_GroupedRatio_{m}"] = cm['grouped_ratio']
            row[f"Feat_Team_MaxPairDist_{m}"] = cm['max_pair_dist']

        # ---- PLAYER POSITION METRICS (Feat_) for 14-25 window
        pos_metrics = compute_player_group_side_metrics(timeline, info['participants'], focus_team_id,
                                                        start_min=14, end_min=25, near_dist=2000.0)
        for role in roles_order:
            pid = role_to_pid.get(role)
            if not pid:
                continue
            pm = pos_metrics.get(pid, {})
            row[f"Feat_{role}_SideRatio_14_25"] = pm.get('side_ratio')
            row[f"Feat_{role}_GroupRatio_14_25"] = pm.get('group_ratio')
            row[f"Feat_{role}_PosValidMin_14_25"] = pm.get('valid_minutes')

        # ---- PLAYER MVP LABELS (Y_)
        for role in roles_order:
            derive_player_mvp_labels(
                row,
                role,
                side_ratio=row.get(f"Feat_{role}_SideRatio_14_25"),
                group_ratio=row.get(f"Feat_{role}_GroupRatio_14_25"),
            )

        # ---- JUNGLE FOCUS (Feat_ + Label_)
        jg = compute_jungle_presence_and_focus(timeline, info['participants'], focus_team_id, start_min=2, end_min=14)
        row['Feat_Jungle_LanePresenceTop_2_14'] = jg['lane_top']
        row['Feat_Jungle_LanePresenceMid_2_14'] = jg['lane_mid']
        row['Feat_Jungle_LanePresenceBot_2_14'] = jg['lane_bot']
        row['Label_JungleFocusLane_2_14'] = jg['lane_focus']

        row['Feat_Jungle_TopsidePresence_2_14'] = jg['side_top']
        row['Feat_Jungle_BotsidePresence_2_14'] = jg['side_bot']
        row['Label_JungleMapSideFocus_2_14'] = jg['side_focus']

        # ---- ROLE-SPECIFIC STYLE LABELS (MVP)
        derive_jungle_gank_labels(row)

        roam_ratio, bot_ratio, vmin = compute_support_roam_ratio(timeline, info['participants'], focus_team_id)
        derive_support_roam_labels(row, roam_ratio, bot_ratio, vmin)

        # ---- TEAM EVENTS FEATURES (Feat_)
        row['Feat_FirstDragonTime_s'] = obj[focus_team_id]['first_dragon']
        row['Feat_FirstGrubsTime_s'] = obj[focus_team_id]['first_grubs']
        row['Feat_FirstHeraldTime_s'] = obj[focus_team_id]['first_herald']
        row['Feat_FirstTowerTime_s'] = towers[focus_team_id]['0_14']['first_tower_time']

        row['Feat_Dragons_0_14'] = obj[focus_team_id]['drag_0_14']
        row['Feat_Grubs_0_14'] = obj[focus_team_id]['grubs_0_14']
        row['Feat_Heralds_0_14'] = obj[focus_team_id]['herald_0_14']
        row['Feat_Objectives_0_14'] = row['Feat_Dragons_0_14'] + row['Feat_Grubs_0_14'] + row['Feat_Heralds_0_14']

        # ---- TEAM MVP LABEL (Y_): early map focus
        derive_team_focus_label(row)
        row['Feat_Kills_0_14'] = fights['kills_0_14'][focus_team_id]
        row['Feat_Kills_14_25'] = fights['kills_14_25'][focus_team_id]

        f014 = fights['fights']['0_14'][focus_team_id]
        row['Feat_Fights_0_14'] = f014['total_fights']
        row['Feat_PickRatio_0_14'] = f014['pick_ratio']
        row['Feat_TeamfightRatio_0_14'] = f014['teamfight_ratio']
        row['Feat_AvgKillsPerFight_0_14'] = f014['avg_kills_per_fight']
        row['Feat_AvgUniqParticipants_0_14'] = f014['avg_uniq_participants']
        row['Feat_FirstFightTime_0_14_s'] = f014['first_fight_time']
        row['Feat_FirstFightType_0_14'] = f014['first_fight_type']

        f1425 = fights['fights']['14_25'][focus_team_id]
        row['Feat_Fights_14_25'] = f1425['total_fights']
        row['Feat_PickRatio_14_25'] = f1425['pick_ratio']
        row['Feat_TeamfightRatio_14_25'] = f1425['teamfight_ratio']
        row['Feat_AvgKillsPerFight_14_25'] = f1425['avg_kills_per_fight']
        row['Feat_AvgUniqParticipants_14_25'] = f1425['avg_uniq_participants']

        f025 = fights['fights']['0_25'][focus_team_id]
        row['Feat_Fights_0_25'] = f025['total_fights']
        row['Feat_PickRatio_0_25'] = f025['pick_ratio']
        row['Feat_TeamfightRatio_0_25'] = f025['teamfight_ratio']
        row['Feat_AvgKillsPerFight_0_25'] = f025['avg_kills_per_fight']
        row['Feat_AvgUniqParticipants_0_25'] = f025['avg_uniq_participants']

        for lane in ('TOP', 'MID', 'BOT'):
            row[f"Feat_TowerKills_{lane}_0_14"] = towers[focus_team_id]['0_14']['towers'][lane]
            row[f"Feat_TowerKills_{lane}_14_25"] = towers[focus_team_id]['14_25']['towers'][lane]
            row[f"Feat_Plates_{lane}_0_14"] = towers[focus_team_id]['0_14']['plates'][lane]

        # ---- LABELS (Y)
        cluster_ctx = {'Team_GroupedRatio_15': row.get('Feat_Team_GroupedRatio_15')}
        row.update(derive_team_labels(focus_team_id, obj, towers, fights, cluster_ctx))

        # ---- Team totals (Post_)
        t_stats = next(t for t in info['teams'] if t['teamId'] == focus_team_id)
        objs = t_stats.get('objectives', {})

        row['Post_Team_Dragons'] = objs.get('dragon', {}).get('kills', 0)
        row['Post_Team_Barons'] = objs.get('baron', {}).get('kills', 0)
        row['Post_Team_Towers'] = objs.get('tower', {}).get('kills', 0)
        row['Post_Team_Inhibitors'] = objs.get('inhibitor', {}).get('kills', 0)
        row['Post_Team_RiftHeralds'] = objs.get('riftHerald', {}).get('kills', 0)
        row['Post_Team_VoidGrubs'] = objs.get('horde', {}).get('kills', 0)

        sample = next((p for p in info['participants'] if p['teamId'] == focus_team_id), None)
        row['Post_Team_ElderDragons'] = sample.get('challenges', {}).get('teamElderDragonKills', 0) if sample else 0

        rows_generated.append(row)

    return rows_generated


def _main_test():
    try:
        acc = riot_call_with_retry(account_watcher.account.by_riot_id, REGION, GAME_NAME, TAG_LINE)
        target_puuid = acc.get('puuid')
    except ApiError as err:
        print(f"ERROR: No se pudo encontrar el PUUID para {GAME_NAME}#{TAG_LINE}: {err}")
        target_puuid = None

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
            import os
            os.makedirs('Data', exist_ok=True)
            pd.DataFrame(dataset).to_csv('Data/dataset_test.csv', index=False)
            print('Prueba finalizada con éxito.')
    else:
        print('No hay PUUID; abortando.')


if __name__ == '__main__':
    _main_test()
