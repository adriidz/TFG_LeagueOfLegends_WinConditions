import os
import pandas as pd
import time
from dotenv import load_dotenv
from riotwatcher import LolWatcher, ApiError

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

def get_match_ids(puuid, count=5, ranked_only=False):
    """Obtiene los IDs usando RiotWatcher (maneja 429 automáticamente)."""
    queue = 420 if ranked_only else None
    try:
        return watcher.match.matchlist_by_puuid(REGION, puuid, count=count, queue=queue)
    except ApiError as err:
        print(f"Error obteniendo IDs: {err}")
        return []

def calculate_early_stats(ally_id, role, frame_15, participant_map, enemy_team_id):
    """
    Calcula la diferencia de oro/xp/cs al minuto 15.
    LÓGICA ORIGINAL CONSERVADA EXACTAMENTE.
    """
    # 1. Datos del Aliado
    if str(ally_id) not in frame_15['participantFrames']: 
        return 0, 0, 0
        
    my_data = frame_15['participantFrames'][str(ally_id)]
    
    # 2. Buscar al Enemigo del mismo Rol
    enemy_stats = {'gold': 0, 'xp': 0, 'cs': 0}
    found = False
    
    for pid, p_info in participant_map.items():
        if p_info['teamId'] == enemy_team_id and p_info['role'] == role:
            if str(pid) in frame_15['participantFrames']:
                en_data = frame_15['participantFrames'][str(pid)]
                enemy_stats['gold'] = en_data['totalGold']
                enemy_stats['xp'] = en_data['xp']
                # Tu fórmula original: minions + jungle
                enemy_stats['cs'] = en_data['minionsKilled'] + en_data['jungleMinionsKilled']
                found = True
            break
            
    if not found: return 0, 0, 0 

    # 3. Calcular Diferencia (Aliado - Enemigo)
    g_diff = my_data['totalGold'] - enemy_stats['gold']
    x_diff = my_data['xp'] - enemy_stats['xp']
    c_diff = (my_data['minionsKilled'] + my_data['jungleMinionsKilled']) - enemy_stats['cs']
    
    return g_diff, x_diff, c_diff

def process_match(match_id):
    try:
        # Descarga Detalles y Timeline con RiotWatcher
        details = watcher.match.by_id(REGION, match_id)
        timeline = watcher.match.timeline_by_match(REGION, match_id)
    except ApiError as err:
        if err.response.status_code == 404:
            print(f"Partida {match_id} no encontrada.")
        else:
            print(f"Error bajando {match_id}: {err}")
        return None

    # --- 1. DATOS GENERALES ---
    info = details['info']
    duration = info['gameDuration'] / 60 # En minutos
    
    # Filtros originales
    if duration < 15 or info.get('gameMode') != 'CLASSIC': 
        return None 

    # Equipo ganador
    teams = info['teams']
    winning_team = 100 if teams[0]['win'] else 200
    losing_team = 200 if winning_team == 100 else 100
    
    # --- 2. MAPEO DE JUGADORES ---
    pmap = {} 
    for p in info['participants']:
        pmap[p['participantId']] = {
            'teamId': p['teamId'],
            'role': p['teamPosition'], 
            'championId': p['championId'],
            'stats': p 
        }

    # --- 3. EXTRACCIÓN DE DATOS ---
    row = {
        'matchId': match_id,
        'win': 1, 
        'gameDuration': duration
    }

    # Frame del minuto 15
    frames = timeline['info']['frames']
    idx = 15 if len(frames) > 15 else len(frames) - 1
    frame_15 = frames[idx]

    roles_order = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
    
    for role in roles_order:
        ally_p = None
        enemy_p = None
        
        # Buscar protagonistas
        for pid, data in pmap.items():
            if data['role'] == role:
                if data['teamId'] == winning_team: ally_p = data
                else: enemy_p = data
        
        if not ally_p or not enemy_p: continue 

        prefix = role 
        # A. INPUTS (IDs)
        row[f"{prefix}_Ally_ID"] = ally_p['championId']
        row[f"{prefix}_Enemy_ID"] = enemy_p['championId']
        
        # B. TARGETS INDIVIDUALES (Early - Minuto 15)
        # Usamos tu función auxiliar original
        g_diff, x_diff, c_diff = calculate_early_stats(
            ally_p['stats']['participantId'], role, frame_15, pmap, losing_team
        )
        row[f"{prefix}_GoldDiff15"] = g_diff
        row[f"{prefix}_XpDiff15"] = x_diff
        row[f"{prefix}_CsDiff15"] = c_diff 

        # C. TARGETS INDIVIDUALES (Globales - Normalizados)
        p_stats = ally_p['stats']
        
        # Mantenemos EXACTAMENTE tus campos originales
        row[f"{prefix}_DmgChamp"] = p_stats['totalDamageDealtToChampions'] / duration
        row[f"{prefix}_DmgTurret"] = p_stats['damageDealtToTurrets'] / duration
        row[f"{prefix}_DmgObj"] = p_stats['damageDealtToObjectives'] / duration
        
        row[f"{prefix}_DmgMitigated"] = p_stats['damageSelfMitigated'] / duration
        row[f"{prefix}_DmgTaken"] = p_stats['totalDamageTaken'] / duration
        
        row[f"{prefix}_Vision"] = p_stats['visionScore'] / duration
        row[f"{prefix}_WardsPlaced"] = p_stats['wardsPlaced'] / duration
        
        row[f"{prefix}_Heal"] = p_stats['totalHeal'] / duration
        row[f"{prefix}_HealsAlly"] = p_stats['totalHealsOnTeammates'] / duration
        row[f"{prefix}_ShieldsAlly"] = p_stats['totalDamageShieldedOnTeammates'] / duration

        row[f"{prefix}_TimeCC"] = p_stats['timeCCingOthers'] / duration
        row[f"{prefix}_TotalCC"] = p_stats['totalTimeCCDealt'] / duration
        
        row[f"{prefix}_KDA_Kills"] = p_stats['kills']
        row[f"{prefix}_KDA_Deaths"] = p_stats['deaths']
        row[f"{prefix}_KDA_Assists"] = p_stats['assists']

    # --- 4. TARGETS GLOBALES DE EQUIPO ---
    # Buscamos el objeto del equipo ganador
    t_stats = next(t for t in info['teams'] if t['teamId'] == winning_team)
    objs = t_stats['objectives']

    row['Team_Dragons'] = objs['champion']['kills'] # Nota: API v5 usa 'champion' para dragones a veces, o 'dragon'
    # En tu código original ponía objs['dragon']. RiotWatcher devuelve el JSON standard.
    # Verificamos: MatchV5 standard usa 'dragon'. Si tu código funcionaba, mantenemos 'dragon'.
    # Si falla, es porque RiotWatcher devuelve 'dragon' o 'champion' segun version. 
    # Usaré .get('dragon', ...) por seguridad.
    
    row['Team_Dragons'] = objs.get('dragon', {}).get('kills', 0)
    row['Team_Barons'] = objs.get('baron', {}).get('kills', 0)
    row['Team_Towers'] = objs.get('tower', {}).get('kills', 0)
    row['Team_Inhibitors'] = objs.get('inhibitor', {}).get('kills', 0)
    row['Team_RiftHeralds'] = objs.get('riftHerald', {}).get('kills', 0)
    row['Team_VoidGrubs'] = objs.get('horde', {}).get('kills', 0) # Horde = VoidGrubs

    # Elder Dragons (desde challenges del jugador, igual que tenías)
    winner_sample = next((p for p in info['participants'] if p['teamId'] == winning_team), None)
    if winner_sample:
        row['Team_ElderDragons'] = winner_sample.get('challenges', {}).get('teamElderDragonKills', 0)
    else:
        row['Team_ElderDragons'] = 0

    return row

def get_my_puuid():
    """Obtiene PUUID por Riot ID usando RiotWatcher."""
    try:
        account = watcher.account.by_riot_id(REGION, GAME_NAME, TAG_LINE)
        print(f"Tu PUUID es: {account['puuid'][:15]}...")
        return account['puuid']
    except ApiError as err:
        print(f"Error buscando usuario: {err}")
        return None

if __name__ == "__main__":
    target_puuid = get_my_puuid()
    if target_puuid:
        print("Iniciando prueba de extracción...")
        match_ids = get_match_ids(target_puuid, count=3, ranked_only=True)
        dataset = []
        for m in match_ids:
            r = process_match(m)
            if r: dataset.append(r)
        
        if dataset:
            pd.DataFrame(dataset).to_csv("test_dataset.csv", index=False)
            print("Prueba finalizada con éxito.")