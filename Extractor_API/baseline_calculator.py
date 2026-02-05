import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split

# --- CONFIGURACIÓN ---
INPUT_FILE = "Data/dataset_train_2.csv"
OUTPUT_JSON = "Data/champion_stats_3.json"
TRAIN_CSV = "Data/train_split.csv"
TEST_CSV = "Data/test_split.csv"

MIN_GAMES_THRESHOLD = 20  # Backoff: si hay menos de 20 games, usamos media del Rol

def generate_robust_baselines():
    print(f"1. Cargando {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("ERROR: No encuentro el archivo. ¿Has ejecutado dataset_transformer.py?")
        return

    # --- CORRECCIÓN CRÍTICA: SPLIT POR MATCH_ID ---
    if 'matchId' not in df.columns:
        print("ERROR CRÍTICO: No encuentro la columna 'matchId'. No puedo hacer el split seguro.")
        # Si tu transformer borró el matchId, tendrás que modificarlo para que lo mantenga.
        return

    print("   Identificando partidas únicas...")
    unique_matches = df['matchId'].unique()
    print(f"   Total de partidas únicas: {len(unique_matches)}")
    
    # Dividimos los IDs de las partidas, no las filas
    train_ids, test_ids = train_test_split(unique_matches, test_size=0.2, random_state=42)
    
    # Filtramos el DataFrame usando esos IDs
    df_train = df[df['matchId'].isin(train_ids)].copy()
    df_test = df[df['matchId'].isin(test_ids)].copy()
    
    # Guardamos los splits para que train.py los use
    df_train.to_csv(TRAIN_CSV, index=False)
    df_test.to_csv(TEST_CSV, index=False)
    
    print(f"-> Split realizado: {len(train_ids)} partidas a Train, {len(test_ids)} a Test.")
    print(f"-> Filas Train: {len(df_train)} | Filas Test: {len(df_test)}")

    # --- CÁLCULO DE BASELINES (SOLO CON TRAIN) ---
    target_cols = [c for c in df.columns if c.startswith('Target_')]
    
    # A) Stats Globales por Rol (Red de seguridad)
    print("2. Calculando Backoff (Stats por Rol)...")
    role_stats = df_train.groupby('Input_Role')[target_cols].agg(['mean', 'std'])
    
    # B) Stats por Campeón
    print("3. Calculando Stats por Campeón...")
    champ_grouped = df_train.groupby(['Input_Player_ID', 'Input_Role'])[target_cols]
    champ_stats_agg = champ_grouped.agg(['mean', 'std'])
    champ_counts = champ_grouped.size()
    
    final_dict = {}
    
    for (champ_id, role), _ in champ_grouped.groups.items():
        key = f"{champ_id}_{role}"
        final_dict[key] = {}
        
        n_games = champ_counts.loc[(champ_id, role)]
        c_row = champ_stats_agg.loc[(champ_id, role)]
        
        # Obtenemos fallback del rol
        try:
            r_row = role_stats.loc[role]
        except KeyError:
            r_row = None
            
        final_dict[key]["games"] = int(n_games)
        final_dict[key]["used_fallback"] = False

        for metric in target_cols:
            c_mean = c_row[(metric, 'mean')]
            c_std = c_row[(metric, 'std')]
            
            # Lógica de Backoff
            use_fallback = False
            if n_games < MIN_GAMES_THRESHOLD:
                use_fallback = True
            elif pd.isna(c_std) or c_std == 0:
                use_fallback = True
            
            if use_fallback and r_row is not None:
                final_mean = r_row[(metric, 'mean')]
                final_std = r_row[(metric, 'std')]
                final_dict[key]["used_fallback"] = True
            else:
                final_mean = c_mean
                final_std = c_std if not pd.isna(c_std) and c_std > 0 else 1.0

            final_dict[key][metric] = {
                "mean": float(final_mean),
                "std": float(final_std)
            }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_dict, f, indent=4)
        
    print(f"-> JSON generado con éxito: {OUTPUT_JSON}")

if __name__ == "__main__":
    generate_robust_baselines()