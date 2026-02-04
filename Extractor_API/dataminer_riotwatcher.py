import os
import pandas as pd
import time
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
                if killer_id > 0: takedowns_counter[killer_id] += 1
                
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
                    'takedowns': takedowns_counter[pid]         
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
            'takedowns': takedowns_counter[pid]
        }
        
    for m in minutes_to_track:
        if m not in snapshots:
            snapshots[m] = last_known_data

    return snapshots

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
    duration = info['gameDuration'] / 60 
    
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

    # Mapear participantes
    pmap = {p['participantId']: p for p in info['participants']}
    rows_generated = []

    for focus_team_id in [100, 200]:

        did_win = 1 if focus_team_id == winning_team_id else 0
        enemy_team_id = 200 if focus_team_id == 100 else 100

        row = {
            'matchId': match_id,
            'teamId': focus_team_id,
            'win': did_win, 
            'gameDuration': duration
        }

        roles_order = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
        valid_team = True

        # Iterar por roles (Perspectiva Equipo Ganador)
        for role in roles_order:
            ally_p = None
            enemy_p = None
            
            for pid, p in pmap.items():
                if p['teamPosition'] == role:
                    if p['teamId'] == focus_team_id: ally_p = p
                    else: enemy_p = p
            
            if not ally_p or not enemy_p:
                valid_team = False
                break
            
            prefix = role 
            pid_ally = ally_p['participantId']
            pid_enemy = enemy_p['participantId']

            # A. IDENTIFICADORES
            row[f"{prefix}_Ally_ID"] = ally_p['championId']
            row[f"{prefix}_Enemy_ID"] = enemy_p['championId']
            
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

            # D. ESTADÍSTICAS GLOBALES
            stats = ally_p
            challenges = ally_p.get('challenges', {})
            
            # Daño
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
                        # row[f"{prefix}_GoldSpent"] = stats['goldSpent']
            
            # Snowball Tracking 
            # bountyGold: Oro obtenido al cobrar recompensas de enemigos
                        # row[f"{prefix}_BountyGold"] = challenges.get('bountyGold', 0)
                        # row[f"{prefix}_FirstBloodKill"] = 1 if stats.get('firstBloodKill', False) else 0
                        # row[f"{prefix}_FirstBloodAssist"] = 1 if stats.get('firstBloodAssist', False) else 0

            # Objective Participation 
            # Takedowns incluye Kills + Asistencias en el objetivo
            row[f"{prefix}_DragonTakedowns"] = challenges.get('dragonTakedowns', 0)
            row[f"{prefix}_BaronTakedowns"] = challenges.get('baronTakedowns', 0)
            row[f"{prefix}_RiftHeraldTakedowns"] = challenges.get('riftHeraldTakedowns', 0)
            # VoidMonsterKill suele referirse a larvas (Void Grubs) o Heraldo dependiendo del patch
            row[f"{prefix}_VoidMonsterTakedowns"] = challenges.get('voidMonsterKill', 0)

            # Desafíos Varios
            row[f"{prefix}_TurretPlates"] = challenges.get('turretPlatesTaken', 0)
            row[f"{prefix}_MaxCsAdvantage"] = challenges.get('maxCsAdvantageOnLaneOpponent', 0)
            row[f"{prefix}_SoloKills"] = challenges.get('soloKills', 0)
            
            # Visión
            row[f"{prefix}_VisionScore"] = stats['visionScore'] / duration
            row[f"{prefix}_WardsPlaced"] = stats['wardsPlaced'] / duration
            row[f"{prefix}_WardsKilled"] = stats['wardsKilled'] / duration
            row[f"{prefix}_ControlWardsPlaced"] = challenges.get('controlWardsPlaced', 0) / duration
                        #  row[f"{prefix}_DetectorWardsPlaced"] = stats.get('detectorWardsPlaced', 0) / duration

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

        # --- 4. TARGETS EQUIPO ---
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

if __name__ == "__main__":
    target_puuid = account_watcher.account.by_riot_id(REGION, GAME_NAME, TAG_LINE).get('puuid')
    if target_puuid:
        print(f"Iniciando prueba de extracción para {GAME_NAME}#{TAG_LINE}...")
        match_ids = get_match_ids(target_puuid, count=3, ranked_only=True)
        dataset = []
        for i,m in enumerate(match_ids):
            print(f"[{i+1}/{len(match_ids)}] ", end="")

            rows = process_match(m)
            if rows: dataset.extend(rows)
            time.sleep(1.3)
        
        if dataset:
            pd.DataFrame(dataset).to_csv("Data/dataset_adriidz.csv", index=False)
            print("Prueba finalizada con éxito.")
    else:
        print(f"ERROR: No se pudo encontrar el PUUID para {GAME_NAME}#{TAG_LINE}. Verifica el nombre, región y la clave API.")