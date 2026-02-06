"""
inference.py — Win Condition Coach: predicción interpretable para un draft.
Combina baseline campeón-rol + ajuste del modelo para dar consejos accionables.
"""
import torch
import json
import numpy as np
from model import LoLWinConditionModel

# --- CONFIG ---
MODEL_PATH = "Models/lol_model_pca_only.pth"
JSON_FILE = "Data/champion_stats_3.json"
CONTEXT_JSON_FILE = "Data/champion_context_pca.json"
CONTEXT_DIM = 6

INDIVIDUAL_METRICS = [
    'GoldDiff_10', 'XpDiff_10', 'CsDiff_10',
    'GoldDiff_15', 'XpDiff_15', 'CsDiff_15',
    'EarlyTakedowns', 'LaneCsBefore10', 'JgCsBefore10',
    'DmgTotal', 'DmgTurret', 'DmgObj', 'VisionScore',
    'KillParticipation', 'SoloKills',
    'TotalCS', 'GoldEarned',
]

TEAM_METRICS = ['Team_Dragons', 'Team_Barons', 'Team_Towers']

STRATEGIC_METRICS = [
    'LaneDominance', 'EarlyAggression', 'MapPressure',
    'ObjectiveFocus', 'ResourceEfficiency',
]

ROLE_MAP = {'TOP': 0, 'JUNGLE': 1, 'MIDDLE': 2, 'BOTTOM': 3, 'UTILITY': 4}

# Nombres legibles para campeones (ejemplos — ampliar)
CHAMP_NAMES = {
    24: 'Jax', 36: 'Mundo', 13: 'Ryze', 523: 'Aphelios', 111: 'Nautilus',
    887: 'Gwen', 59: 'JarvanIV', 134: 'Syndra', 42: 'Corki', 517: 'Sylas',
    64: 'LeeSin', 266: 'Aatrox', 200: 'BelVeth', 245: 'Ekko',
    126: 'Jayce', 92: 'Riven', 58: 'Renekton', 103: 'Ahri',
}

UNITS = {
    'GoldDiff_10': 'gold', 'GoldDiff_15': 'gold',
    'XpDiff_10': 'xp', 'XpDiff_15': 'xp',
    'CsDiff_10': 'cs', 'CsDiff_15': 'cs',
    'EarlyTakedowns': 'kills', 'LaneCsBefore10': 'cs', 'JgCsBefore10': 'cs',
    'DmgTotal': 'dmg/min', 'DmgTurret': 'dmg/min', 'DmgObj': 'dmg/min',
    'VisionScore': 'vs/min', 'KillParticipation': '%',
    'SoloKills': 'kills', 'TotalCS': 'cs', 'GoldEarned': 'gold',
    'Team_Dragons': 'dragons', 'Team_Barons': 'barons', 'Team_Towers': 'towers',
}


def load_model():
    model = LoLWinConditionModel(
        num_individual_metrics=len(INDIVIDUAL_METRICS),
        num_team_metrics=len(TEAM_METRICS),
        num_strategic_metrics=len(STRATEGIC_METRICS),
        mode='pca_only',
        champ_pca_dim=CONTEXT_DIM,
        team_context_dim=CONTEXT_DIM * 2,
    )
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location='cpu'))
    model.eval()
    return model


def predict_for_role(model, role, allies, enemies, pca_context, stats_json, win=1.0):
    """
    allies: dict {role: champion_id} para los 5 roles
    enemies: dict {role: champion_id} para los 5 roles
    role: rol del jugador foco ('TOP', 'JUNGLE', etc.)

    Retorna dict con predicciones desnormalizadas.
    """
    roles_order = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
    empty_ctx = [0.0] * CONTEXT_DIM

    # PCA vectors
    player_id = allies[role]
    player_pca = pca_context.get(f"{player_id}_{role}", empty_ctx)

    allies_pca = []
    enemies_pca = []
    ctx_ally = np.zeros(CONTEXT_DIM, dtype=np.float32)
    ctx_enemy = np.zeros(CONTEXT_DIM, dtype=np.float32)

    for r in roles_order:
        a_vec = np.array(pca_context.get(f"{allies[r]}_{r}", empty_ctx), dtype=np.float32)
        e_vec = np.array(pca_context.get(f"{enemies[r]}_{r}", empty_ctx), dtype=np.float32)
        ctx_ally += a_vec
        ctx_enemy += e_vec
        if r != role:
            allies_pca.append(a_vec.tolist())
        enemies_pca.append(e_vec.tolist())

    team_context = np.concatenate([ctx_ally, ctx_enemy])

    # Tensors
    p = torch.tensor([player_pca], dtype=torch.float32)
    a = torch.tensor([allies_pca], dtype=torch.float32)
    e = torch.tensor([enemies_pca], dtype=torch.float32)
    r_id = torch.tensor([ROLE_MAP[role]], dtype=torch.long)
    w = torch.tensor([[win]], dtype=torch.float32)
    ctx = torch.tensor([team_context], dtype=torch.float32)

    with torch.no_grad():
        pred_ind, pred_team, pred_strat = model(p, a, e, r_id, w, team_context=ctx)

    z_ind = pred_ind[0].numpy()
    z_team = pred_team[0].numpy()
    z_strat = pred_strat[0].numpy()

    # Desnormalizar: predicción_real = mean + z × std
    key = f"{player_id}_{role}"
    champ_stats = stats_json.get(key, {})

    results = {'role': role, 'champion_id': player_id,
               'champion': CHAMP_NAMES.get(player_id, str(player_id))}

    # Individual
    results['individual'] = {}
    for i, m in enumerate(INDIVIDUAL_METRICS):
        col = f"Target_{m}"
        if col in champ_stats:
            mean = champ_stats[col]['mean']
            std = champ_stats[col]['std']
            pred_real = mean + z_ind[i] * std
            results['individual'][m] = {
                'baseline': mean, 'z_adjust': float(z_ind[i]),
                'prediction': pred_real, 'unit': UNITS.get(m, ''),
            }

    # Team
    results['team'] = {}
    for i, m in enumerate(TEAM_METRICS):
        col = f"Target_{m}"
        if col in champ_stats:
            mean = champ_stats[col]['mean']
            std = champ_stats[col]['std']
            pred_real = mean + z_team[i] * std
            results['team'][m] = {
                'baseline': mean, 'z_adjust': float(z_team[i]),
                'prediction': pred_real,
            }

    # Strategic
    results['strategic'] = {}
    for i, m in enumerate(STRATEGIC_METRICS):
        col = f"Target_{m}"
        if col in champ_stats:
            mean = champ_stats[col]['mean']
            std = champ_stats[col]['std']
            pred_real = mean + z_strat[i] * std
            results['strategic'][m] = {
                'baseline': mean, 'z_adjust': float(z_strat[i]),
                'prediction': pred_real,
            }

    return results


def print_coaching(results):
    champ = results['champion']
    role = results['role']

    print(f"\n{'='*60}")
    print(f"  WIN CONDITION COACH — {champ} {role}")
    print(f"{'='*60}")

    # Strategic overview
    print(f"\n  PERFIL ESTRATÉGICO (0 = nulo, 1 = máximo):")
    print(f"  {'Target':<25s} | {'Baseline':>8s} | {'Draft Adj':>9s} | {'Final':>6s}")
    print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*9}-+-{'-'*6}")
    for m, data in results['strategic'].items():
        arrow = '▲' if data['z_adjust'] > 0.1 else ('▼' if data['z_adjust'] < -0.1 else '→')
        print(f"  {m:<25s} | {data['baseline']:8.3f} | {arrow} {data['z_adjust']:+6.2f}σ | {data['prediction']:6.3f}")

    # Key individual metrics
    print(f"\n  MÉTRICAS CLAVE:")
    print(f"  {'Métrica':<20s} | {'Baseline':>10s} | {'Draft Adj':>9s} | {'Predicción':>10s}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*9}-+-{'-'*10}")
    key_metrics = ['GoldDiff_10', 'GoldDiff_15', 'EarlyTakedowns', 'DmgTotal',
                   'DmgTurret', 'DmgObj', 'KillParticipation', 'SoloKills']
    for m in key_metrics:
        if m in results['individual']:
            d = results['individual'][m]
            arrow = '▲' if d['z_adjust'] > 0.1 else ('▼' if d['z_adjust'] < -0.1 else '→')
            print(f"  {m:<20s} | {d['baseline']:>10.1f} | {arrow} {d['z_adjust']:+6.2f}σ | {d['prediction']:>10.1f} {d['unit']}")

    # Team objectives
    print(f"\n  OBJETIVOS DE EQUIPO:")
    for m, d in results['team'].items():
        arrow = '▲' if d['z_adjust'] > 0.1 else ('▼' if d['z_adjust'] < -0.1 else '→')
        print(f"  {m:<20s} | {d['baseline']:>5.1f} | {arrow} {d['z_adjust']:+.2f}σ | {d['prediction']:>5.1f}")

    print()


# === EJEMPLO DE USO ===
if __name__ == "__main__":
    print("Cargando modelo...")
    model = load_model()

    with open(JSON_FILE, 'r') as f:
        stats_json = json.load(f)
    with open(CONTEXT_JSON_FILE, 'r') as f:
        pca_context = json.load(f)

    # Draft ejemplo: mismo de la primera fila del dataset
    allies = {'TOP': 24, 'JUNGLE': 36, 'MIDDLE': 13, 'BOTTOM': 523, 'UTILITY': 111}
    enemies = {'TOP': 887, 'JUNGLE': 59, 'MIDDLE': 134, 'BOTTOM': 42, 'UTILITY': 517}

    for role in ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']:
        results = predict_for_role(
            model, role, allies, enemies,
            pca_context, stats_json, win=0.5)
        print_coaching(results)