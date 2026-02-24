import pandas as pd
import numpy as np

# --- CONFIGURACIÓN ---
INPUT_FILE = "Data/dataset_raw_1.csv"
OUTPUT_FILE = "Data/dataset_full_2.csv"

# Columnas a eliminar del resultado final (antes de renombrar)
COLUMNS_TO_DROP = [
    'Target_LaneCsBefore10',
    'Target_JgCsBefore10',
    'Target_EnemyJgInvades',
]


def transform_to_player_centric(input_csv, output_csv):
    print(f"Cargando {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print("Error: No se encuentra el archivo.")
        return

    print(f"Total de partidas a procesar (Wins + Losses): {len(df)}")

    player_rows = []

    roles = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']

    # Columnas que son de equipo (se mantienen igual para los 5 jugadores)
    team_cols = [c for c in df.columns if c.startswith('Team_')]

    print("Transformando filas a formato player-centric...")

    for idx, row in df.iterrows():
        # Extraemos IDs de Aliados y Enemigos
        context_ids = {}
        for r in roles:
            context_ids[f"Input_Ally_{r}_ID"] = row[f"{r}_Ally_ID"]
            context_ids[f"Input_Enemy_{r}_ID"] = row[f"{r}_Enemy_ID"]

        for target_role in roles:
            new_row = {
                'matchId': row['matchId'],
                'gameDuration': row['gameDuration'],
                'win': row['win'],
                'Input_Role': target_role,
                'Input_Player_ID': row[f"{target_role}_Ally_ID"],
            }

            new_row.update(context_ids)

            # --- TARGETS (Métricas Individuales) ---
            prefix = f"{target_role}_"

            for col in df.columns:
                if col.startswith(prefix) and not col.endswith('_ID'):
                    metric_name = col.replace(prefix, "Target_")
                    new_row[metric_name] = row[col]

            # --- TARGETS (Métricas de Equipo) ---
            for t_col in team_cols:
                new_row[f"Target_{t_col}"] = row[t_col]

            player_rows.append(new_row)

    df_final = pd.DataFrame(player_rows)

    # --- ELIMINAR COLUMNAS NO DESEADAS ---
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df_final.columns]
    if cols_to_drop:
        df_final.drop(columns=cols_to_drop, inplace=True)
        print(f"Columnas eliminadas: {cols_to_drop}")

    # --- RENOMBRAR: Eliminar prefijos Input_ y Target_ ---
    rename_map = {}
    for col in df_final.columns:
        if col.startswith('Input_'):
            rename_map[col] = col[len('Input_'):]
        elif col.startswith('Target_'):
            rename_map[col] = col[len('Target_'):]
    df_final.rename(columns=rename_map, inplace=True)

    # Guardar
    df_final.to_csv(output_csv, index=False)
    print(f"¡Hecho! Generadas {len(df_final)} filas de entrenamiento.")
    print(f"Guardado en: {output_csv}")


if __name__ == "__main__":
    transform_to_player_centric(INPUT_FILE, OUTPUT_FILE)