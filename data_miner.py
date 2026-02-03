import os
import requests
import pandas as pd
import time
from dotenv import load_dotenv

# --- CONFIGURACIÓN ---
load_dotenv('TFG.env')

API_KEY = os.getenv('RIOT_API_KEY')
REGION = os.getenv('REGION', 'europe')
MATCH_REGION = os.getenv('MATCH_REGION', 'euw1')

GAME_NAME = 'adriidz'
TAG_LINE = 'diaz'

# Validación de seguridad
if not API_KEY:
    raise ValueError("ERROR: No se encontró la variable 'RIOT_API_KEY' en el archivo .env")

headers = {"X-Riot-Token": API_KEY}

def get_match_ids(puuid, count=5, ranked_only=False):
    """Obtiene los IDs de las últimas partidas."""
    if ranked_only:    
        url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count={count}&queue=420"
    else:
        url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count={count}"

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            print("  [!] Rate Limit (429). Esperando 14s...")
            time.sleep(14)
            return get_match_ids(puuid, count, ranked_only) # Reintento recursivo
        elif response.status_code == 403:
            print("Error 403")
            return []
        else:
            print(f"Error {response.status_code} obteniendo IDs.")
            return []
    except Exception as e:
        print(f"Error de conexión: {e}")
        return []

def get_match_data(match_id):
    """Descarga Detalles y Timeline a la vez."""
    url_det = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    url_time = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    
    # Función auxiliar para manejar reintentos
    def make_request(url):
        while True:
            resp = requests.get(url, headers=headers)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get('Retry-After', 14)) # Riot a veces dice cuánto esperar
                print(f"  [!] Rate Limit en MatchData. Pausa de {retry_after}s...")
                time.sleep(retry_after)
                continue # Volvemos a intentar
            return resp
    
    res_det = make_request(url_det)

    # Pequeña pausa para no saturar el Rate Limit si no tenemos clave de producción
    time.sleep(0.1) 
    
    res_time = make_request(url_time)
    
    if res_det.status_code == 200 and res_time.status_code == 200:
        return res_det.json(), res_time.json()
    
    print(f"Error bajando datos de {match_id}. Status: D={res_det.status_code}, T={res_time.status_code}")
    return None, None

def calculate_early_stats(ally_id, role, frame_15, participant_map, enemy_team_id):
    """Calcula la diferencia de oro/xp/cs al minuto 15."""
    
    # 1. Datos del Aliado
    # Protección: A veces el participantId no está en el frame (bugs raros de la API o desconexiones)
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
    print(f"Procesando {match_id}...")
    details, timeline = get_match_data(match_id)
    
    if not details or not timeline: return None
    
    # --- 1. DATOS GENERALES ---
    info = details['info']
    duration = info['gameDuration'] / 60 # En minutos
    
    # Ignorar remakes (menos de 15 min) o partidas ARAM/Arena si se colaran
    if duration < 15 or info.get('gameMode') != 'CLASSIC': 
        return None 

    winning_team = 100 if info['teams'][0]['win'] else 200
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

    # Frame del minuto 15 (o el último disponible si acabó antes, aunque filtramos <15)
    frames = timeline['info']['frames']
    idx = 15 if len(frames) > 15 else len(frames) - 1
    frame_15 = frames[idx]

    roles_order = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
    
    for role in roles_order:
        ally_p = None
        enemy_p = None
        
        # Buscar protagonistas de la línea
        for pid, data in pmap.items():
            if data['role'] == role:
                if data['teamId'] == winning_team: ally_p = data
                else: enemy_p = data
        
        # Si la API no asignó roles correctamente (pasa en partidas raras), saltamos
        if not ally_p or not enemy_p: continue 

        prefix = role 
        # A. INPUTS (IDs)
        row[f"{prefix}_Ally_ID"] = ally_p['championId']
        row[f"{prefix}_Enemy_ID"] = enemy_p['championId']
        
        # B. TARGETS INDIVIDUALES (Early - Minuto 15)
        g_diff, x_diff, c_diff = calculate_early_stats(
            ally_p['stats']['participantId'], role, frame_15, pmap, losing_team
        )
        row[f"{prefix}_GoldDiff15"] = g_diff
        row[f"{prefix}_XpDiff15"] = x_diff
        row[f"{prefix}_CsDiff15"] = c_diff 

        # C. TARGETS INDIVIDUALES (Globales - Normalizados por minuto)
        p_stats = ally_p['stats']
        
        # Ofensivas
        row[f"{prefix}_DmgChamp"] = p_stats['totalDamageDealtToChampions'] / duration
        row[f"{prefix}_DmgTurret"] = p_stats['damageDealtToTurrets'] / duration
        row[f"{prefix}_DmgObj"] = p_stats['damageDealtToObjectives'] / duration
        
        # Defensivas
        row[f"{prefix}_DmgMitigated"] = p_stats['damageSelfMitigated'] / duration
        row[f"{prefix}_DmgTaken"] = p_stats['totalDamageTaken'] / duration
        
        # Utilidad / Mapa
        row[f"{prefix}_Vision"] = p_stats['visionScore'] / duration
        row[f"{prefix}_WardsPlaced"] = p_stats['wardsPlaced'] / duration
        
        # Curación y apoyo
        row[f"{prefix}_Heal"] = p_stats['totalHeal'] / duration
        row[f"{prefix}_HealsAlly"] = p_stats['totalHealsOnTeammates'] / duration
        row[f"{prefix}_ShieldsAlly"] = p_stats['totalDamageShieldedOnTeammates'] / duration

        # Control de masas
        row[f"{prefix}_TimeCC"] = p_stats['timeCCingOthers'] / duration
        row[f"{prefix}_TotalCC"] = p_stats['totalTimeCCDealt'] / duration
        
        # KDA (Para filtrado posterior)
        row[f"{prefix}_KDA_Kills"] = p_stats['kills']
        row[f"{prefix}_KDA_Deaths"] = p_stats['deaths']
        row[f"{prefix}_KDA_Assists"] = p_stats['assists']

    # --- 4. TARGETS GLOBALES DE EQUIPO ---
    t_stats = info['teams'][0] if info['teams'][0]['teamId'] == winning_team else info['teams'][1]
    objs = t_stats['objectives']

    row['Team_Dragons'] = objs['dragon']['kills']
    row['Team_Barons'] = objs['baron']['kills']
    row['Team_Towers'] = objs['tower']['kills']
    row['Team_Inhibitors'] = objs['inhibitor']['kills']
    row['Team_RiftHeralds'] = objs['riftHerald']['kills']
    row['Team_VoidGrubs'] = objs.get('horde', {}).get('kills', 0)

    # Sacamos el número de Elder Dragons
    # Buscamos un participante cualquiera del equipo ganador
    winner_sample = None
    for p in info['participants']:
        if p['teamId'] == winning_team:
            winner_sample = p
            break
            
    if winner_sample:
        # Usamos .get() porque en partidas muy antiguas o modos raros 'challenges' podría no estar
        challenges = winner_sample.get('challenges', {})
        row['Team_ElderDragons'] = challenges.get('teamElderDragonKills', 0)
    else:
        row['Team_ElderDragons'] = 0

    return row

def get_my_puuid(headers):
    """
    PASO 1: Convertir tu Nombre Humano en el ID de Robot (PUUID).
    Endpoint usado: ACCOUNT-V1
    """
    # Construimos la URL. Fíjate que usamos 'riot' y 'account'.
    url = f"https://{REGION}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{GAME_NAME}/{TAG_LINE}"
    
    print(f"--- Consultando a Riot: ¿Quién es {GAME_NAME}#{TAG_LINE}? ---")
    
    # Hacemos la llamada (GET)
    response = requests.get(url, headers=headers)
    
    # 200 significa "OK, todo ha ido bien".
    if response.status_code == 200:
        data = response.json() # Convertimos la respuesta en un diccionario de Python
        puuid = data['puuid']
        print(f"Tu PUUID es: {puuid[:15]}...")
        return puuid
    else:
        # Si falla (ej: 403 Forbidden, 404 Not Found), avisamos.
        print(f"Error buscando usuario: Código {response.status_code}")
        return None

# --- EJECUCIÓN ---
if __name__ == "__main__":
    # PUUID de un jugador de prueba (ej: Agurin, Caps, etc.)
    target_puuid = get_my_puuid(headers=headers)
    
    if not target_puuid:
        print("ALERTA: PUUID no válido.")
    else:
        print("Iniciando extracción de datos...")
        match_ids = get_match_ids(target_puuid, count=3, ranked_only=True)
        dataset = []
        
        for i, m in enumerate(match_ids):
            print(f"[{i+1}/{len(match_ids)}] ", end="")
            r = process_match(m)
            if r: dataset.append(r)
            time.sleep(1.3) # Respetamos límites API
            
        if dataset:
            df = pd.DataFrame(dataset)
            filename = "dataset_adriidz.csv"
            df.to_csv(filename, index=False)
            print(f"\nÉXITO: {len(df)} partidas guardadas en {filename}")
        else:
            print("\nNo se pudieron procesar partidas. Revisa el PUUID o la API Key.")