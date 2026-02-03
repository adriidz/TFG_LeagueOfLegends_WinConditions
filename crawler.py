import os
import requests
import pandas as pd
import time
from dotenv import load_dotenv

# Importamos tus funciones del otro script
# Asegúrate de que data_miner.py esté en la misma carpeta
from data_miner import process_match, get_match_ids, headers, REGION, MATCH_REGION

load_dotenv('TFG.env')

# --- CONFIGURACIÓN DEL CRAWLER ---
MATCHES_PER_PLAYER = 10  # Bajar pocas partidas por jugador para tener más variedad
TOTAL_MATCHES_TARGET = 10000 # Objetivo inicial (luego pon 5000)
CSV_FILENAME = "dataset_raw.csv"

def get_challengers_league(retries=3):
    """
    Paso 1: Obtener la lista de los 300 mejores jugadores (Challengers).
    Devuelve una lista de diccionarios con 'summonerId'.
    """
    # Endpoint: Challenger League
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5"
    
    print(f"Descargando la lista de Challengers de {MATCH_REGION}...")
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            entries = response.json()['entries']
            print(f"Se han encontrado {len(entries)} jugadores Challenger.")
            return entries
        elif response.status_code == 429: # RATE LIMIT
            if retries > 0:
                print(f"  Rate Limit (429). Esperando 14s... (Reintentando)")
                time.sleep(14)
                return get_challengers_league(retries=retries-1) # <--- Reintento Recursivo
            else:
                print("  Demasiados intentos fallidos. Abortando.")
                return []
        else:
            print(f"Error {response.status_code} obteniendo liga.")
            return []
    except Exception as e:
        print(f"Error de conexión: {e}")
        return []
    
def get_riot_id(puuid, retries=1):
    """
    Obtiene el GameName#TagLine a partir del PUUID.
    Usa la región 'europe' (REGION), no 'euw1'.
    """
    url = f"https://{REGION}.api.riotgames.com/riot/account/v1/accounts/by-puuid/{puuid}"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return f"{data['gameName']}#{data['tagLine']}"
        elif response.status_code == 429:
            if retries > 0:
                print("  [!] Rate Limit (429). Esperando 14s...")
                time.sleep(14)
                return get_riot_id(puuid, retries=retries-1) # Reintento recursivo
            else:
                return "Unknown#Limit"
        else:
            return "Unknown#???"
    except:
        return "Unknown#???"

def load_existing_matches(filename):
    """Carga los IDs de partidas que ya tenemos para no repetirlas."""
    if not os.path.exists(filename):
        return set()
    df = pd.read_csv(filename)
    # Devuelve un conjunto (set) con los matchId para búsqueda rápida
    return set(df['matchId'].unique())

# --- BUCLE PRINCIPAL ---
if __name__ == "__main__":
    
    # 1. Cargar lo que ya tenemos
    processed_ids = load_existing_matches(CSV_FILENAME)
    print(f"Base de datos actual: {len(processed_ids)} partidas.")

    # Creamos un set temporal para IDs descartados en esta sesión (para no re-intentarlos en el mismo bucle)
    session_ignored_ids = set()
    
    # 2. Conseguir la lista de Pros
    challengers = get_challengers_league()
    # Ordenamos por LP para coger a los mejores de los mejores primero
    challengers.sort(key=lambda x: x['leaguePoints'], reverse=True)
    
    new_data_buffer = []
    matches_collected = 0
    
    # 3. Iterar por cada Pro Player
    for i, player in enumerate(challengers):
        if matches_collected >= TOTAL_MATCHES_TARGET:
            print("Objetivo alcanzado!")
            break
            
        puuid = player.get('puuid')
        
        # Si por lo que sea esta entrada no tiene puuid, la saltamos
        if not puuid:
            print(f"Saltando jugador {i+1} (Sin PUUID)...")
            continue
            
        print(f"\nAnalizando ({i+1}/{len(challengers)}): {get_riot_id(puuid)}...") # Nombre real
        
        # B. Obtener sus partidas recientes
        match_ids = get_match_ids(puuid, count=MATCHES_PER_PLAYER, ranked_only=True)
        
        for m_id in match_ids:
            # C. Verificar duplicados (¡Muy importante!)
            if m_id in processed_ids or m_id in session_ignored_ids:
                print(f"  Existing -> {m_id} (Saltar)")
                continue
            
            # D. Procesar Partida
            match_data = process_match(m_id)
            
            if match_data:
                new_data_buffer.append(match_data)
                processed_ids.add(m_id)
                matches_collected += 1
                print(f"  Guardada ({matches_collected}/{TOTAL_MATCHES_TARGET})")
            else:
                session_ignored_ids.add(m_id)
                print(f"  Fallida -> {m_id} (Ignorar)")
            
            # Rate Limit Saver (API Dev Key: 20 requests / 1 seg)
            time.sleep(1.3) 
            
            if matches_collected >= TOTAL_MATCHES_TARGET:
                break
        
        # Guardado intermedio cada vez que acabamos con un jugador (por si se cuelga)
        if new_data_buffer:
            df_new = pd.DataFrame(new_data_buffer)
            # Modo 'append': Si el archivo existe, no escribimos la cabecera
            write_header = not os.path.exists(CSV_FILENAME)
            df_new.to_csv(CSV_FILENAME, mode='a', header=write_header, index=False)
            print(f"Guardado parcial de {len(new_data_buffer)} filas.")
            new_data_buffer = [] # Limpiamos buffer

    print("\nProceso finalizado.")