import pandas as pd
import numpy as np

# --- CONFIGURACIÓN ---
INPUT_FILE = "Data/dataset_raw_1.csv"  # Tu archivo actual
OUTPUT_FILE = "Data/dataset_full_2.csv" # El archivo para la IA

def transform_to_player_centric(input_csv, output_csv):
    print(f"Cargando {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print("Error: No se encuentra el archivo.")
        return

    # 1. FILTRAR SOLO VICTORIAS (Behavioral Cloning)
    # Si quisieras validar con derrotas, comenta esta línea.
    # df_wins = df[df['win'] == 1].copy()
    print(f"Total de partidas a procesar (Wins + Losses): {len(df)}")
    # print(f"Partidas ganadoras (para entrenar): {len(df_wins)}")

    player_rows = []
    
    roles = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
    
    # Columnas que son de equipo (se mantienen igual para los 5 jugadores)
    team_cols = [c for c in df.columns if c.startswith('Team_')]
    
    print("Transformando filas... (Esto puede tardar un poco)")
    
    for idx, row in df.iterrows():
        # Para cada partida, generamos 5 filas (una por jugador)
        
        # Extraemos IDs de Aliados y Enemigos para el contexto global
        # (Esto es común para los 5 jugadores de este equipo)
        context_ids = {}
        for r in roles:
            context_ids[f"Input_Ally_{r}_ID"] = row[f"{r}_Ally_ID"]
            context_ids[f"Input_Enemy_{r}_ID"] = row[f"{r}_Enemy_ID"]
            
        for target_role in roles:
            new_row = {
                'matchId': row['matchId'],
                'gameDuration': row['gameDuration'],
                'win': row['win'],
                # --- INPUTS (Contexto) ---
                'Input_Role': target_role,
                'Input_Player_ID': row[f"{target_role}_Ally_ID"], # El ID del protagonista
            }
            
            # Añadimos los IDs de todos (Contexto de Draft)
            new_row.update(context_ids)
            
            # --- TARGETS (Métricas Individuales) ---
            # Buscamos todas las columnas que empiecen por "ROL_" (ej: TOP_GoldDiff_15)
            # y las renombramos a "Target_GoldDiff_15"
            prefix = f"{target_role}_"
            
            for col in df.columns:
                if col.startswith(prefix) and not col.endswith('_ID'):
                    # Quitamos el prefijo del rol (TOP_) y ponemos Target_
                    metric_name = col.replace(prefix, "Target_")
                    new_row[metric_name] = row[col]
            
            # --- TARGETS (Métricas de Equipo) ---
            for t_col in team_cols:
                new_row[f"Target_{t_col}"] = row[t_col]
            
            player_rows.append(new_row)

    # Convertir a DataFrame
    df_final = pd.DataFrame(player_rows)
    
    # Guardar
    df_final.to_csv(output_csv, index=False)
    print(f"¡Hecho! Generadas {len(df_final)} filas de entrenamiento.")
    print(f"Guardado en: {output_csv}")
    
    # # Verificación rápida
    # print("\nColumnas generadas:")
    # print(df_final.columns[:].tolist())

if __name__ == "__main__":
    transform_to_player_centric(INPUT_FILE, OUTPUT_FILE)