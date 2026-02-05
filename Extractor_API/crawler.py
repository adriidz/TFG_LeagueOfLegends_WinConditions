import os
import pandas as pd
import time
import random
from dotenv import load_dotenv
from riotwatcher import LolWatcher, RiotWatcher, ApiError

# Importamos funciones de data_miner
from dataminer import process_match, API_KEY, REGION, MATCH_REGION

load_dotenv('TFG.env')

# --- CONFIGURACIÓN ---
MATCHES_PER_PLAYER = 20 
TOTAL_MATCHES_TARGET = 20000 
CSV_FILENAME = "Data/dataset_raw_1.csv"

# Inicializamos Watcher
watcher = LolWatcher(API_KEY)
account_watcher = RiotWatcher(API_KEY)

def get_high_elo_players():
    """Descarga y combina Challengers, Grandmasters y Masters."""
    challengers = []
    grandmasters = []
    masters = []
    queue_type = 'RANKED_SOLO_5x5'
    
    print(f"--- Recopilando Jugadores de Élite ({MATCH_REGION}) ---")

    # 1. CHALLENGER
    try:
        print("1. Descargando Challenger...", end=" ")
        c_league = watcher.league.challenger_by_queue(MATCH_REGION, queue_type)
        for e in c_league['entries']:
            e["tier"] = c_league.get('tier', 'CHALLENGER')  # Añadimos el rango al entry
        challengers = c_league['entries']
        print(f"OK ({len(c_league['entries'])} jugadores)")
    except ApiError as err:
        print(f"Error Challenger: {err}")

    # 2. GRANDMASTER
    try:
        print("2. Descargando Grandmaster...", end=" ")
        gm_league = watcher.league.grandmaster_by_queue(MATCH_REGION, queue_type)
        for e in gm_league['entries']:
            e["tier"] = gm_league.get('tier', 'GRANDMASTER')  # Añadimos el rango al entry
        grandmasters = gm_league['entries']
        print(f"OK ({len(gm_league['entries'])} jugadores)")
    except ApiError as err:
        print(f"Error Grandmaster: {err}")

    # 3. MASTER
    try:
        print("3. Descargando Master...", end=" ")
        m_league = watcher.league.masters_by_queue(MATCH_REGION, queue_type)
        for e in m_league['entries']:
            e["tier"] = m_league.get('tier', 'MASTER')  # Añadimos el rango al entry
        masters = m_league['entries']
        print(f"OK ({len(m_league['entries'])} jugadores)")
    except ApiError as err:
        print(f"Error Master: {err}")

    print(f"-> Total jugadores encontrados: {len(challengers) + len(grandmasters) + len(masters)}")
    
    # 2. Barajar cada liga individualmente (para no coger siempre el Top 1)
    random.shuffle(challengers)
    random.shuffle(grandmasters)
    random.shuffle(masters)

    # 3. INTERCALADO (ROUND ROBIN)
    # Esto crea una lista: [C_1, GM_1, M_1, C_2, GM_2, M_2, ...]
    balanced_list = []
    
    # Mientras quede alguien en alguna lista...
    while challengers or grandmasters or masters:
        if challengers:
            balanced_list.append(challengers.pop())
        if grandmasters:
            balanced_list.append(grandmasters.pop())
        if masters:
            balanced_list.append(masters.pop())
    
    return balanced_list

def get_match_ids_safe(puuid, count):
    """Wrapper seguro para obtener IDs."""
    try:
        return watcher.match.matchlist_by_puuid(REGION, puuid, count=count, queue=420)
    except ApiError:
        return []

# --- BUCLE PRINCIPAL ---
def run_crawler(get_name=False):
    # 1. Obtener lista combinada de jugadores
    challenger_entries = get_high_elo_players()

    processed_ids = set()
    session_ignored_ids = set()
    matches_collected = 0

    # Cargar IDs ya procesados si existe el CSV (para no repetir si reinicias)
    if os.path.exists(CSV_FILENAME):
        try:
            df_existing = pd.read_csv(CSV_FILENAME)
            if 'matchId' in df_existing.columns:
                processed_ids = set(df_existing['matchId'].unique())
                print(f"CSV existente cargado. Saltando {len(processed_ids)} partidas ya guardadas.")
                matches_collected = len(processed_ids) # Para seguir la cuenta real
        except Exception as e:
            print(f"Nota: No se pudo leer CSV existente o estaba vacío: {e}")

    # Buffer para guardar en bloques (menos escrituras en disco)
    new_data_buffer = []

    for idx, entry in enumerate(challenger_entries):
        if matches_collected >= TOTAL_MATCHES_TARGET:
            print("¡Meta de partidas alcanzada!")
            break
        
        # Obtener PUUID y nombre si es necesario
        puuid = entry.get('puuid')

        if not puuid:
            print(f"  Saltando jugador {idx} (Sin PUUID)")
            continue

        if get_name:
            try:
                acct = account_watcher.account.by_puuid(REGION, puuid)
                name = f"{acct['gameName']}#{acct['tagLine']}"
                tier = entry.get('tier', 'Unknown')
            except Exception as e:
                name = f"Unknown#{idx}"  # Fallback con índice para debug
                print(f"  [!] Error obteniendo nombre: {e}")
        else:
            name = f"Player_{idx}"

        print(f"\nAnalizando ({idx+1}/{len(challenger_entries)}): {name} (Rank: {tier})")

        match_ids = get_match_ids_safe(puuid, MATCHES_PER_PLAYER)
        
        for m_id in match_ids:
            # CHEQUEO DE DUPLICADOS (Ahorro de API)
            if m_id in processed_ids or m_id in session_ignored_ids:
                # print(f"  Saltando repetida: {m_id}") 
                continue
            
            print(f"  Procesando {m_id}...", end="")
            rows = process_match(m_id) # Usa data_miner actualizado
            
            if rows:
                new_data_buffer.extend(rows)
                processed_ids.add(m_id)
                matches_collected += 1
                print(f" OK. (2 filas) Total: {matches_collected}")
            else:
                session_ignored_ids.add(m_id)
                print(f" Ignorada.")
            
            # Guardado parcial cada X partidas o al cambiar de jugador
            if matches_collected >= TOTAL_MATCHES_TARGET:
                break
            
            # Pequeño sleep para ser amable, aunque RiotWatcher gestiona el 429
            time.sleep(0.8)

        # Volcado a disco tras cada jugador
        if new_data_buffer:
            df = pd.DataFrame(new_data_buffer)
            header = not os.path.exists(CSV_FILENAME)
            df.to_csv(CSV_FILENAME, mode='a', header=header, index=False)
            new_data_buffer = [] # Limpiar buffer
            print(f"  -> Guardado parcial en CSV {CSV_FILENAME} (Total partidas: {matches_collected})")

if __name__ == "__main__":
    run_crawler(get_name=True)