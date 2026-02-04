import pandas as pd
import json

# --- CONFIGURACIÓN ---
INPUT_FILE = "Data/dataset_final_train.csv"
OUTPUT_JSON = "Data/champion_stats.json"  # Diccionario para la IA
OUTPUT_CSV = "Data/champion_stats.csv"    # Para ti (Excel)

# Lista de métricas que queremos (las "USEFUL_METRICS" del paso anterior)
# Si no filtras aquí, calculará todo lo que empiece por Target_
def calculate_baselines():
    print(f"Cargando {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Error: No se encuentra el archivo. Ejecuta primero dataset_transformer.py")
        return

    # 1. Seleccionar columnas numéricas (Targets)
    target_cols = [c for c in df.columns if c.startswith('Target_')]
    print(f"Métricas detectadas: {len(target_cols)}")

    # 2. Agrupar por ID y Rol
    # Calculamos Media (mean) y Desviación Estándar (std) a la vez
    grouped = df.groupby(['Input_Player_ID', 'Input_Role'])[target_cols]
    
    # Aggregation: media y std
    df_stats = grouped.agg(['mean', 'std'])
    
    # Añadimos conteo de partidas (para saber si el dato es fiable)
    counts = grouped.size()

    # Correción por si acaso hay NaNs (desviación estándar de un solo dato)
    df_stats = df_stats.fillna(1.0)
    
    # 3. Guardar CSV (Aplanamos las columnas para que sea legible en Excel)
    # Las columnas quedarán tipo: Target_GoldDiff_15_mean, Target_GoldDiff_15_std
    df_flat = df_stats.copy()
    df_flat.columns = ['_'.join(col).strip() for col in df_flat.columns.values]
    df_flat['Games_Played'] = counts
    df_flat.to_csv(OUTPUT_CSV)
    print(f"-> CSV guardado: {OUTPUT_CSV}")

    # 4. Guardar JSON (Estructura optimizada para carga rápida en Python)
    # Clave: "CHAMPID_ROLE" -> Valor: { "Target_Gold": {"mean": 100, "std": 20}, ... }
    
    stats_dict = {}
    
    # Iteramos sobre los grupos
    for (champ_id, role), group_indices in grouped.groups.items():
        key = f"{champ_id}_{role}"
        stats_dict[key] = {}
        
        # Para ese grupo, sacamos los valores calculados
        # df_stats tiene índice MultiIndex (Id, Rol) y Columnas MultiIndex (Métrica, Stat)
        row = df_stats.loc[(champ_id, role)]
        
        for metric in target_cols:
            mean_val = row[(metric, 'mean')]
            std_val = row[(metric, 'std')]

            if pd.isna(std_val):
                std_val = 1.0  # Evitar NaNs

            elif std_val == 0:
                std_val = 1e-6  # Evitar división por cero

            stats_dict[key][metric] = {
                "mean": float(mean_val),
                "std": float(std_val)
            }
        
        stats_dict[key]["games"] = int(counts.loc[(champ_id, role)])

    # Guardamos el JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(stats_dict, f, indent=4)
        
    print(f"-> JSON guardado: {OUTPUT_JSON}")
    print("\n[INFO] Ahora tienes la media y la desviación estándar para calcular Z-Scores.")

if __name__ == "__main__":
    calculate_baselines()