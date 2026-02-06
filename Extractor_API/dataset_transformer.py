import pandas as pd
import numpy as np

# --- CONFIGURACIÓN ---
INPUT_FILE = "Data/dataset_raw_1.csv"
OUTPUT_FILE = "Data/dataset_full_2.csv"


def compute_strategic_targets_absolute(df_players, global_stats):
    """
    Calcula 5 targets estratégicos a partir de valores ABSOLUTOS (no z-normalizados).
    Cada componente se escala a [0, 1] usando percentiles globales para que
    las ponderaciones funcionen correctamente entre métricas de escalas distintas.
    
    global_stats: dict[col_name] -> {'p5': float, 'p95': float}
    """
    results = []
    
    for idx, row in df_players.iterrows():
        role = row['Input_Role']
        
        def norm(col_name):
            """Normalización min-max robusta usando percentiles 5-95 globales."""
            val = row.get(col_name, 0.0)
            stats = global_stats.get(col_name, {})
            p5 = stats.get('p5', 0.0)
            p95 = stats.get('p95', 1.0)
            rng = p95 - p5
            if rng <= 0:
                rng = 1.0
            return np.clip((val - p5) / rng, 0.0, 1.0)
        
        # --- 1. LANE DOMINANCE ---
        # Valores brutos de diferencias de línea (pueden ser negativos → norm los escala)
        if role == 'JUNGLE':
            lane_dom = (
                0.30 * norm('Target_GoldDiff_10') +
                0.20 * norm('Target_XpDiff_10') +
                0.25 * norm('Target_JgCsBefore10') +
                0.15 * norm('Target_GoldDiff_15') +
                0.10 * norm('Target_SoloKills')
            )
        else:
            lane_dom = (
                0.25 * norm('Target_GoldDiff_10') +
                0.20 * norm('Target_XpDiff_10') +
                0.20 * norm('Target_CsDiff_10') +
                0.15 * norm('Target_GoldDiff_15') +
                0.10 * norm('Target_SoloKills') +
                0.10 * norm('Target_TurretPlates')
            )
        
        # --- 2. EARLY AGGRESSION ---
        early_agg = (
            0.40 * norm('Target_EarlyTakedowns') +
            0.30 * norm('Target_SoloKills') +
            0.30 * norm('Target_KillParticipation')
        )
        
        # --- 3. MAP PRESSURE ---
        map_press = (
            0.30 * norm('Target_VisionScore') +
            0.35 * norm('Target_KillParticipation') +
            0.20 * norm('Target_WardsPlaced') +
            0.15 * norm('Target_ControlWardsPlaced')
        )
        
        # --- 4. OBJECTIVE FOCUS ---
        obj_focus = (
            0.25 * norm('Target_DmgObj') +
            0.25 * norm('Target_DmgTurret') +
            0.20 * norm('Target_DragonTakedowns') +
            0.15 * norm('Target_BaronTakedowns') +
            0.15 * norm('Target_TurretPlates')
        )
        
        # --- 5. RESOURCE EFFICIENCY ---
        # DPG (damage per gold) como ratio directo
        gold = row.get('Target_GoldEarned', 1)
        dmg = row.get('Target_DmgTotal', 0)
        dpg_raw = dmg / gold if gold > 0 else 0
        dpg_stats = global_stats.get('_dpg', {'p5': 0, 'p95': 1})
        dpg_rng = dpg_stats['p95'] - dpg_stats['p5']
        if dpg_rng <= 0:
            dpg_rng = 1.0
        dpg_norm = np.clip((dpg_raw - dpg_stats['p5']) / dpg_rng, 0.0, 1.0)
        
        res_eff = (
            0.40 * dpg_norm +
            0.30 * norm('Target_TotalCS') +
            0.30 * norm('Target_GoldEarned')
        )
        
        results.append({
            'Target_LaneDominance': float(lane_dom),
            'Target_EarlyAggression': float(early_agg),
            'Target_MapPressure': float(map_press),
            'Target_ObjectiveFocus': float(obj_focus),
            'Target_ResourceEfficiency': float(res_eff),
        })
    
    return pd.DataFrame(results)


def compute_global_percentile_stats(df_players):
    """
    Calcula percentiles 5 y 95 de cada métrica Target_ usada en targets estratégicos.
    Esto es global (no por rol ni por campeón) para capturar escalas absolutas.
    """
    cols_needed = [
        'Target_GoldDiff_10', 'Target_XpDiff_10', 'Target_CsDiff_10',
        'Target_GoldDiff_15', 'Target_XpDiff_15', 'Target_CsDiff_15',
        'Target_JgCsBefore10', 'Target_SoloKills', 'Target_TurretPlates',
        'Target_EarlyTakedowns', 'Target_KillParticipation',
        'Target_VisionScore', 'Target_WardsPlaced', 'Target_ControlWardsPlaced',
        'Target_DmgObj', 'Target_DmgTurret',
        'Target_DragonTakedowns', 'Target_BaronTakedowns',
        'Target_TotalCS', 'Target_GoldEarned', 'Target_DmgTotal',
    ]
    
    global_stats = {}
    for col in cols_needed:
        if col in df_players.columns:
            global_stats[col] = {
                'p5': df_players[col].quantile(0.05),
                'p95': df_players[col].quantile(0.95),
            }
        else:
            global_stats[col] = {'p5': 0.0, 'p95': 1.0}
    
    # DPG ratio
    gold = df_players['Target_GoldEarned'].replace(0, np.nan)
    dpg = df_players['Target_DmgTotal'] / gold
    dpg = dpg.dropna()
    global_stats['_dpg'] = {
        'p5': dpg.quantile(0.05),
        'p95': dpg.quantile(0.95),
    }
    
    return global_stats


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
    
    # --- CALCULAR TARGETS ESTRATÉGICOS (ESCALA ABSOLUTA) ---
    print("Calculando targets estratégicos (escala absoluta)...")
    
    global_stats = compute_global_percentile_stats(df_final)
    strategic_df = compute_strategic_targets_absolute(df_final, global_stats)
    df_final = pd.concat([df_final, strategic_df], axis=1)
    
    # Verificación rápida
    for col in ['Target_LaneDominance', 'Target_EarlyAggression', 'Target_MapPressure',
                'Target_ObjectiveFocus', 'Target_ResourceEfficiency']:
        print(f"  {col}: mean={df_final[col].mean():.3f}, std={df_final[col].std():.3f}, "
              f"min={df_final[col].min():.3f}, max={df_final[col].max():.3f}")
    
    # Guardar
    df_final.to_csv(output_csv, index=False)
    print(f"¡Hecho! Generadas {len(df_final)} filas de entrenamiento.")
    print(f"Guardado en: {output_csv}")


if __name__ == "__main__":
    transform_to_player_centric(INPUT_FILE, OUTPUT_FILE)