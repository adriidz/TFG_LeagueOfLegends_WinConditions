import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split

# --- CONFIGURACIÓN ---
INPUT_FILE = "Data/dataset_full_2.csv"
OUTPUT_JSON = "Data/champion_stats_3.json"
TRAIN_CSV = "Data/train_split.csv"
TEST_CSV = "Data/test_split.csv"

MIN_GAMES_THRESHOLD = 20

def generate_robust_baselines():
    print(f"1. Cargando {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("ERROR: No encuentro el archivo.")
        return

    # Limpieza previa: Rellenar NaNs en el DataFrame original con 0 para evitar propagación
    # Esto es útil si alguna columna venía vacía del crawler
    df.fillna(0, inplace=True)

    if 'matchId' not in df.columns:
        print("ERROR CRÍTICO: No encuentro la columna 'matchId'.")
        return

    print("   Identificando partidas únicas...")
    unique_matches = df['matchId'].unique()
    train_ids, test_ids = train_test_split(unique_matches, test_size=0.2, random_state=42)
    
    df_train = df[df['matchId'].isin(train_ids)].copy()
    df_test = df[df['matchId'].isin(test_ids)].copy()
    
    df_train.to_csv(TRAIN_CSV, index=False)
    df_test.to_csv(TEST_CSV, index=False)
    
    # --- CÁLCULO DE BASELINES ---
    target_cols = [c for c in df.columns if c.startswith('Target_')]
    
    print("2. Calculando Stats por Rol (Fallback)...")
    # Calculamos media y std por ROL (ej: todos los TOPs juntos)
    role_stats = df_train.groupby('Input_Role')[target_cols].agg(['mean', 'std'])
    
    print("3. Calculando Stats por Campeón...")
    champ_grouped = df_train.groupby(['Input_Player_ID', 'Input_Role'])[target_cols]
    champ_stats_agg = champ_grouped.agg(['mean', 'std'])
    champ_counts = champ_grouped.size()
    
    final_dict = {}
    
    for (champ_id, role), _ in champ_grouped.groups.items():
        key = f"{champ_id}_{role}"
        final_dict[key] = {}
        
        n_games = champ_counts.loc[(champ_id, role)]
        final_dict[key]["games"] = int(n_games)
        
        # Recuperamos la fila de estadísticas del ROL por si la necesitamos
        try:
            r_row = role_stats.loc[role]
        except KeyError:
            r_row = None

        # Fila del campeón específico
        c_row = champ_stats_agg.loc[(champ_id, role)]

        for metric in target_cols:
            # 1. Intentamos obtener valores del campeón
            val_mean = c_row[(metric, 'mean')]
            val_std = c_row[(metric, 'std')]
            
            # 2. Condiciones para usar Fallback (Rol)
            use_fallback = False
            
            # A) Pocas partidas
            if n_games < MIN_GAMES_THRESHOLD:
                use_fallback = True
            
            # B) Valores corruptos (NaN o Infinito)
            if pd.isna(val_mean) or pd.isna(val_std) or np.isinf(val_mean) or np.isinf(val_std):
                use_fallback = True
                
            # C) Desviación estándar 0 (Evita división por cero luego)
            if val_std == 0:
                use_fallback = True

            # 3. Aplicar Lógica
            final_mean = 0.0
            final_std = 1.0
            
            if use_fallback and r_row is not None:
                # Usamos la media del ROL
                r_mean = r_row[(metric, 'mean')]
                r_std = r_row[(metric, 'std')]
                
                # Si el rol también está mal (muy raro), ponemos 0 y 1
                final_mean = float(r_mean) if not pd.isna(r_mean) else 0.0
                final_std = float(r_std) if not pd.isna(r_std) and r_std > 0 else 1.0
                
            else:
                # Usamos la media del CAMPEÓN
                final_mean = float(val_mean)
                final_std = float(val_std) if val_std > 0 else 1.0

            final_dict[key][metric] = {
                "mean": final_mean,
                "std": final_std
            }

    # Guardado seguro
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_dict, f, indent=4)
        
    print(f"-> JSON generado SIN NaNs: {OUTPUT_JSON}")

if __name__ == "__main__":
    generate_robust_baselines()