import os
import pandas as pd
import time
import random
from dotenv import load_dotenv
from riotwatcher import LolWatcher, RiotWatcher, ApiError

# Importamos funciones de data_miner
from dataminer_v3 import process_match, API_KEY, REGION, MATCH_REGION

load_dotenv('TFG.env')

# --- CONFIGURACIÓN ---
MATCHES_PER_PLAYER = 80 
TOTAL_MATCHES_TARGET = 80000
CSV_FILENAME = "Data/dataset_raw_v3.csv"
SAVE_EVERY_N_MATCHES = 50  # Guardar cada N partidas para no perder todo en caso de error

# Inicializamos Watcher
watcher = LolWatcher(API_KEY)
account_watcher = RiotWatcher(API_KEY)

def get_high_elo_players():
    """Descarga y combina Challengers, Grandmasters y Masters."""
    queue_type = 'RANKED_SOLO_5x5'
    tiers = []

    print(f"--- Recopilando Jugadores de Élite ({MATCH_REGION}) ---")

    for name, func in [("Challenger", watcher.league.challenger_by_queue),
                        ("Grandmaster", watcher.league.grandmaster_by_queue),
                        ("Master", watcher.league.masters_by_queue)]:
        try:
            league = func(MATCH_REGION, queue_type)
            for e in league['entries']:
                e["tier"] = league.get('tier', name.upper())
            tiers.append(league['entries'])
            print(f"  {name}: {len(league['entries'])} jugadores")
        except ApiError as err:
            print(f"  {name}: Error - {err}")
            tiers.append([])

    # Intercalado round-robin
    for t in tiers:
        random.shuffle(t)

    balanced = []
    while any(tiers):
        for t in tiers:
            if t:
                balanced.append(t.pop())

    print(f"-> Total: {len(balanced)} jugadores")
    return balanced

def get_match_ids_safe(puuid, count):
    """Wrapper seguro para obtener IDs."""
    try:
        return watcher.match.matchlist_by_puuid(REGION, puuid, count=count, queue=420)
    except ApiError:
        return []

def load_existing_schema(csv_path):
    """Carga el esquema (orden de columnas) del CSV existente, si existe."""
    if not os.path.exists(csv_path):
        return None
    try:
        # nrows=0 -> solo cabecera
        existing_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        return existing_cols if existing_cols else None
    except Exception as e:
        print(f"Nota: No se pudo leer el esquema del CSV existente: {e}")
        return None


def align_to_schema(df, schema_cols):
    """Reindexa df al esquema. Si aparecen columnas nuevas, se descartan para no romper el CSV existente."""
    if schema_cols is None:
        return df, df.columns.tolist(), []
    extra = [c for c in df.columns if c not in schema_cols]
    if extra:
        print(f"AVISO: Se han detectado columnas nuevas no presentes en el CSV y se descartarán: {extra}")
    df_aligned = df.reindex(columns=schema_cols)
    return df_aligned, schema_cols, extra


def append_csv_consistent(csv_path, df, schema_cols):
    """Append a CSV garantizando orden de columnas estable."""
    header = not os.path.exists(csv_path)
    if header:
        schema_cols = df.columns.tolist()
        df.to_csv(csv_path, mode='a', header=True, index=False)
        return schema_cols

    df_aligned, schema_cols, _ = align_to_schema(df, schema_cols)
    df_aligned.to_csv(csv_path, mode='a', header=False, index=False)
    return schema_cols


# --- BUCLE PRINCIPAL ---
def run_crawler():
    # 1. Obtener lista combinada de jugadores
    players = get_high_elo_players()

    processed_ids = set()
    session_ignored_ids = set()
    matches_collected = 0

    # Esquema estable de columnas para evitar CSV inconsistente
    schema_cols = load_existing_schema(CSV_FILENAME)

    # Cargar IDs ya procesados si existe el CSV (para no repetir si reinicias)
    if os.path.exists(CSV_FILENAME):
        try:
            df_existing = pd.read_csv(CSV_FILENAME, usecols=['matchId'])
            processed_ids = set(df_existing['matchId'].unique())
            matches_collected = len(processed_ids) # Para seguir la cuenta real
            print(f"CSV existente cargado. Reanudando: {len(processed_ids)} partidas ya guardadas.")
        except Exception as e:
            print(f"Nota: No se pudo leer CSV existente o estaba vacío: {e}")

    # Buffer para guardar en bloques (menos escrituras en disco)
    new_data_buffer = []
    unsaved_count = 0

    for idx, entry in enumerate(players):
        if matches_collected >= TOTAL_MATCHES_TARGET:
            print("¡Meta de partidas alcanzada!")
            break
        
        # Obtener PUUID y nombre si es necesario
        puuid = entry.get('puuid')

        if not puuid:
            print(f"  Saltando jugador {idx} (Sin PUUID)")
            continue

        tier = entry.get('tier', '?')
        print(f"\n[{idx+1}/{len(players)}] Rank: {tier} | Total: {matches_collected}")

        match_ids = get_match_ids_safe(puuid, MATCHES_PER_PLAYER)

        # Filtrar duplicados ANTES de hacer API calls
        new_ids = [m for m in match_ids
                   if m not in processed_ids and m not in session_ignored_ids]
        
        if not new_ids:
            print("Sin partidas nuevas para este jugador.")
            continue
            
        for m_id in new_ids:
            if matches_collected >= TOTAL_MATCHES_TARGET:
                break

            print(f"  Procesando {m_id}...", end="")
            rows = process_match(m_id)
        
            if rows:
                new_data_buffer.extend(rows)
                processed_ids.add(m_id)
                matches_collected += 1
                unsaved_count += 1
                print(f"OK. Total: {matches_collected}")
            else:
                session_ignored_ids.add(m_id)
                print(f" Ignorada.")
        
            # Pequeño sleep para ser amable, aunque RiotWatcher gestiona el 429
            time.sleep(0.3)
        
        if unsaved_count >= SAVE_EVERY_N_MATCHES and new_data_buffer:
            df = pd.DataFrame(new_data_buffer)
            schema_cols = append_csv_consistent(CSV_FILENAME, df, schema_cols)
            new_data_buffer = []  # Limpiar buffer
            unsaved_count = 0
            print(f"  -> Guardado parcial en CSV {CSV_FILENAME} (Total partidas: {matches_collected})")

    # Volcado a disco tras cada jugador
    if new_data_buffer:
        df = pd.DataFrame(new_data_buffer)
        schema_cols = append_csv_consistent(CSV_FILENAME, df, schema_cols)
        new_data_buffer = []  # Limpiar buffer
        print(f"  -> Guardado parcial en CSV {CSV_FILENAME} (Total partidas: {matches_collected})")

if __name__ == "__main__":
    run_crawler()