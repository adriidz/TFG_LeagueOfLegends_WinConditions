import os
import pandas as pd
import time
from dotenv import load_dotenv
from riotwatcher import LolWatcher, RiotWatcher, ApiError

# Importamos funciones de data_miner
from dataminer_riotwatcher import process_match, API_KEY, REGION, MATCH_REGION

load_dotenv('TFG.env')

# --- CONFIGURACIÓN ---
MATCHES_PER_PLAYER = 10 
TOTAL_MATCHES_TARGET = 10000 
CSV_FILENAME = "Data/dataset_raw_riotwatcher.csv"

# Inicializamos Watcher
watcher = LolWatcher(API_KEY)
account_watcher = RiotWatcher(API_KEY)

def get_challengers_league():
    """Obtiene la lista de Challengers de forma segura."""
    print(f"Descargando lista Challengers de {MATCH_REGION}...")
    try:
        league = watcher.league.challenger_by_queue(MATCH_REGION, 'RANKED_SOLO_5x5')
        entries = league['entries']
        print(f"Encontrados {len(entries)} jugadores.")
        return entries
    except ApiError as err:
        print(f"Error obteniendo liga: {err}")
        return []

def get_match_ids_safe(puuid, count):
    """Wrapper seguro para obtener IDs."""
    try:
        return watcher.match.matchlist_by_puuid(REGION, puuid, count=count, queue=420)
    except ApiError:
        return []

def load_existing_matches(filename):
    if not os.path.exists(filename):
        return set()
    try:
        df = pd.read_csv(filename)
        return set(df['matchId'].unique())
    except:
        return set()

# --- BUCLE PRINCIPAL ---
def run_crawler(get_name=False):
    processed_ids = load_existing_matches(CSV_FILENAME)
    print(f"Dataset actual: {len(processed_ids)} partidas.")

    session_ignored_ids = set()
    
    challengers = get_challengers_league()
    # Ordenamos por LP
    challengers.sort(key=lambda x: x['leaguePoints'], reverse=True)
    
    new_data_buffer = []
    matches_collected = len(processed_ids)
    
    for i, player in enumerate(challengers):
        if matches_collected >= TOTAL_MATCHES_TARGET:
            print("¡Objetivo alcanzado!")
            break
        
        # Obtener PUUID y nombre si es necesario
        puuid = player.get('puuid')

        if not puuid:
            print(f"  Saltando jugador {i} (Sin PUUID)")
            continue

        if get_name:
            try:
                acct = account_watcher.account.by_puuid(REGION, puuid)
                name = f"{acct['gameName']}#{acct['tagLine']}"
            except Exception as e:
                name = f"Unknown#{i}"  # Fallback con índice para debug
                print(f"  [!] Error obteniendo nombre: {e}")
        else:
            name = f"Player_{i}"

        print(f"\nAnalizando ({i+1}/{len(challengers)}): {name}")

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
            time.sleep(0.5)

        # Volcado a disco tras cada jugador
        if new_data_buffer:
            df = pd.DataFrame(new_data_buffer)
            header = not os.path.exists(CSV_FILENAME)
            df.to_csv(CSV_FILENAME, mode='a', header=header, index=False)
            new_data_buffer = [] # Limpiar buffer
            print("  -> Guardado parcial en CSV.")

if __name__ == "__main__":
    run_crawler(get_name=True)