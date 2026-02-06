import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json
import numpy as np
import os
from collections import defaultdict
from model import LoLWinConditionModel
from loss import LoLTripleLoss

# --- CONFIGURACIÓN ---
BATCH_SIZE = 256
LEARNING_RATE = 0.0005
EPOCHS = 100
EMBEDDING_DIM = 32
CONTEXT_DIM = 6  # PCA components por campeón

EARLY_STOPPING_PATIENCE = 12
DIAGNOSTIC_EVERY = 5

TRAIN_FILE = "Data/train_split.csv"
TEST_FILE = "Data/test_split.csv"
JSON_FILE = "Data/champion_stats_3.json"
CONTEXT_JSON_FILE = "Data/champion_context_pca.json"

INDIVIDUAL_METRICS = [
    'GoldDiff_10', 'XpDiff_10', 'CsDiff_10',
    'GoldDiff_15', 'XpDiff_15', 'CsDiff_15',
    'EarlyTakedowns', 'LaneCsBefore10', 'JgCsBefore10',
    'DmgTotal', 'DmgTurret', 'DmgObj', 'VisionScore',
    'KillParticipation', 'SoloKills',
    'TotalCS', 'GoldEarned',
]

TEAM_METRICS = [
    'Team_Dragons', 'Team_Barons', 'Team_Towers',
]

STRATEGIC_METRICS = [
    'LaneDominance', 'EarlyAggression', 'MapPressure',
    'ObjectiveFocus', 'ResourceEfficiency',
]

# Unidades legibles para cada métrica raw
UNITS = {
    'GoldDiff_10': 'gold', 'GoldDiff_15': 'gold',
    'XpDiff_10': 'xp', 'XpDiff_15': 'xp',
    'CsDiff_10': 'cs', 'CsDiff_15': 'cs',
    'EarlyTakedowns': 'kills', 'LaneCsBefore10': 'cs', 'JgCsBefore10': 'cs',
    'DmgTotal': 'dmg/min', 'DmgTurret': 'dmg/min', 'DmgObj': 'dmg/min',
    'VisionScore': 'vs/min', 'KillParticipation': '%',
    'SoloKills': 'kills', 'TotalCS': 'cs', 'GoldEarned': 'gold',
    'Team_Dragons': 'drags', 'Team_Barons': 'barons', 'Team_Towers': 'towers',
}


def compute_avg_std_per_metric(stats_json):
    """Calcula la std media por métrica Target_ a través de todos los champion-role."""
    std_accum = defaultdict(list)
    for champ_key, metrics in stats_json.items():
        for metric_key, metric_data in metrics.items():
            if isinstance(metric_data, dict) and 'std' in metric_data:
                std_accum[metric_key].append(metric_data['std'])
    return {k: np.mean(v) for k, v in std_accum.items()}


# =====================================================================
#  DATASET
# =====================================================================
class LoLDataset(Dataset):
    def __init__(self, csv_file, json_file, individual_metrics, team_metrics,
                 strategic_metrics, context_json, experiment_mode='pca_only', mode='train'):
        df_raw = pd.read_csv(csv_file)
        self.data = df_raw.copy().reset_index(drop=True)
        self.experiment_mode = experiment_mode

        n_wins = (self.data['win'] == 1).sum()
        n_losses = (self.data['win'] == 0).sum()
        print(f"Dataset ({mode}): {len(self.data)} filas ({n_wins} wins, {n_losses} losses)")

        with open(json_file, 'r') as f:
            self.stats = json.load(f)

        self.individual_metrics = individual_metrics
        self.team_metrics = team_metrics
        self.strategic_metrics = strategic_metrics
        self.all_raw_metrics = individual_metrics + team_metrics
        self.n_individual = len(individual_metrics)
        self.n_team = len(team_metrics)
        self.n_strategic = len(strategic_metrics)

        self.role_map = {'TOP': 0, 'JUNGLE': 1, 'MIDDLE': 2, 'BOTTOM': 3, 'UTILITY': 4}

        with open(context_json, 'r') as f:
            self.pca_context = json.load(f)

        self.context_dim = CONTEXT_DIM
        self.empty_context = [0.0] * self.context_dim

    def get_team_context_pca(self, team_tuples):
        team_vector = np.zeros(self.context_dim, dtype=np.float32)
        for cid, role in team_tuples:
            vec = self.pca_context.get(f"{cid}_{role}", self.empty_context)
            team_vector += np.array(vec, dtype=np.float32)
        return team_vector

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        player_id = int(row['Input_Player_ID'])
        role_str = row['Input_Role']
        role_id = self.role_map.get(role_str, 0)
        win = float(row['win'])

        roles_order = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
        allies_tuples, enemies_tuples = [], []
        allies_ids, enemies_ids = [], []
        allies_pca, enemies_pca = [], []
        player_pca = None

        for r in roles_order:
            a_id = int(row[f'Input_Ally_{r}_ID'])
            e_id = int(row[f'Input_Enemy_{r}_ID'])
            allies_tuples.append((a_id, r))
            enemies_tuples.append((e_id, r))

            a_pca = self.pca_context.get(f"{a_id}_{r}", self.empty_context)
            e_pca = self.pca_context.get(f"{e_id}_{r}", self.empty_context)

            if r != role_str:
                allies_ids.append(a_id)
                allies_pca.append(a_pca)
            else:
                player_pca = a_pca

            enemies_ids.append(e_id)
            enemies_pca.append(e_pca)

        # Team context agregado
        ctx_ally = self.get_team_context_pca(allies_tuples)
        ctx_enemy = self.get_team_context_pca(enemies_tuples)
        team_context = np.concatenate([ctx_ally, ctx_enemy])

        # --- TARGETS Z-Scores (brutos) ---
        key = f"{player_id}_{role_str}"
        champ_stats = self.stats.get(key, {})
        z_individual, z_team = [], []

        for metric in self.all_raw_metrics:
            col_name = f"Target_{metric}"
            real_val = row.get(col_name, 0.0)
            if col_name in champ_stats:
                mean_g = champ_stats[col_name]["mean"]
                std_g = champ_stats[col_name]["std"]
                z = (real_val - mean_g) / std_g if std_g > 0 else 0.0
                z = max(-4.0, min(4.0, z))
            else:
                z = 0.0
            if metric in self.individual_metrics:
                z_individual.append(float(z))
            else:
                z_team.append(float(z))

       # --- TARGETS ESTRATÉGICOS (z-normalización por campeón-rol, igual que brutos) ---
        z_strategic = []
        for metric in self.strategic_metrics:
            col_name = f"Target_{metric}"
            real_val = row.get(col_name, 0.0)
            if col_name in champ_stats:
                mean_g = champ_stats[col_name]["mean"]
                std_g = champ_stats[col_name]["std"]
                z = (real_val - mean_g) / std_g if std_g > 0 else 0.0
                z = max(-4.0, min(4.0, z))
            else:
                z = 0.0
            z_strategic.append(float(z))

        # --- Construir output ---
        out = {
            'role': torch.tensor(role_id, dtype=torch.long),
            'win': torch.tensor([win], dtype=torch.float32),
            'team_context': torch.tensor(team_context, dtype=torch.float32),
            'target_individual': torch.tensor(z_individual, dtype=torch.float32),
            'target_team': torch.tensor(z_team, dtype=torch.float32),
            'target_strategic': torch.tensor(z_strategic, dtype=torch.float32),
        }

        if self.experiment_mode in ('both', 'emb_only'):
            out['player'] = torch.tensor(player_id, dtype=torch.long)
            out['allies'] = torch.tensor(allies_ids, dtype=torch.long)
            out['enemies'] = torch.tensor(enemies_ids, dtype=torch.long)
        
        if self.experiment_mode == 'pca_only':
            out['player_pca'] = torch.tensor(player_pca, dtype=torch.float32)
            out['allies_pca'] = torch.tensor(allies_pca, dtype=torch.float32)
            out['enemies_pca'] = torch.tensor(enemies_pca, dtype=torch.float32)

        return out


# =====================================================================
#  DIAGNÓSTICO — R², MAE (z-score), MAE (unidades reales)
# =====================================================================
def print_diagnostic(model, loader, individual_metrics, team_metrics,
                     strategic_metrics, device, exp_mode, avg_std=None):
    """
    Métricas interpretables:
      R² = 1 - MSE_modelo / MSE_baseline  → "el draft explica X% de la varianza"
      MAE (z) = error absoluto medio en z-scores
      MAE (real) = MAE(z) × avg_std → error en unidades originales (gold, xp, cs...)
    """
    model.eval()
    all_ind_preds, all_ind_targets = [], []
    all_team_preds, all_team_targets = [], []
    all_strat_preds, all_strat_targets = [], []

    with torch.no_grad():
        for batch in loader:
            p_in, a_in, e_in, ctx = _unpack_model_inputs(batch, device, exp_mode)
            r = batch['role'].to(device)
            w = batch['win'].to(device)

            pred_ind, pred_team, pred_strat = model(p_in, a_in, e_in, r, w, team_context=ctx)

            all_ind_preds.append(pred_ind.cpu())
            all_ind_targets.append(batch['target_individual'])
            all_team_preds.append(pred_team.cpu())
            all_team_targets.append(batch['target_team'])
            all_strat_preds.append(pred_strat.cpu())
            all_strat_targets.append(batch['target_strategic'])

    all_ind_preds = torch.cat(all_ind_preds)
    all_ind_targets = torch.cat(all_ind_targets)
    all_team_preds = torch.cat(all_team_preds)
    all_team_targets = torch.cat(all_team_targets)
    all_strat_preds = torch.cat(all_strat_preds)
    all_strat_targets = torch.cat(all_strat_targets)

    # MSE y MAE por métrica
    mse_model_ind = ((all_ind_preds - all_ind_targets) ** 2).mean(dim=0)
    mse_baseline_ind = (all_ind_targets ** 2).mean(dim=0)
    mae_ind = (all_ind_preds - all_ind_targets).abs().mean(dim=0)

    mse_model_team = ((all_team_preds - all_team_targets) ** 2).mean(dim=0)
    mse_baseline_team = (all_team_targets ** 2).mean(dim=0)
    mae_team = (all_team_preds - all_team_targets).abs().mean(dim=0)

    mse_model_strat = ((all_strat_preds - all_strat_targets) ** 2).mean(dim=0)
    mse_baseline_strat = (all_strat_targets ** 2).mean(dim=0)
    mae_strat = (all_strat_preds - all_strat_targets).abs().mean(dim=0)

    def r2(mse_mod, mse_bas):
        return (1 - mse_mod / mse_bas) * 100 if mse_bas > 0 else 0.0

    has_real = avg_std is not None and len(avg_std) > 0

    # Header
    if has_real:
        print(f"\n  {'Métrica':<25s} | {'R²':>6s} | {'MAE(z)':>7s} | {'MAE(real)':>12s} |")
        sep = f"  {'-'*25}-+-{'-'*6}-+-{'-'*7}-+-{'-'*12}-+"
    else:
        print(f"\n  {'Métrica':<25s} | {'R²':>6s} | {'MAE(z)':>7s} |")
        sep = f"  {'-'*25}-+-{'-'*6}-+-{'-'*7}-+"
    print(sep)

    ok_raw, total_raw = 0, 0

    # --- Individual + Team ---
    for metrics_list, mse_mod, mse_bas, mae_z in [
        (individual_metrics, mse_model_ind, mse_baseline_ind, mae_ind),
        (team_metrics, mse_model_team, mse_baseline_team, mae_team),
    ]:
        for i, m in enumerate(metrics_list):
            total_raw += 1
            is_ok = mse_mod[i].item() < mse_bas[i].item()
            if is_ok:
                ok_raw += 1
            r2_val = r2(mse_mod[i].item(), mse_bas[i].item())
            mae_z_val = mae_z[i].item()
            flag = "OK" if is_ok else "MAL"

            if has_real:
                s = avg_std.get(f"Target_{m}", 0)
                if s > 0:
                    mae_real = mae_z_val * s
                    unit = UNITS.get(m, '')
                    print(f"  {m:<25s} | {r2_val:5.1f}% | {mae_z_val:7.3f} | {mae_real:>8.1f} {unit:<3s} | {flag}")
                else:
                    print(f"  {m:<25s} | {r2_val:5.1f}% | {mae_z_val:7.3f} | {'N/A':>12s} | {flag}")
            else:
                print(f"  {m:<25s} | {r2_val:5.1f}% | {mae_z_val:7.3f} | {flag}")

    # --- Strategic ---
    print(sep)
    print(f"  {'STRATEGIC TARGETS':^25s}")
    print(sep)

    ok_strat = 0
    for i, m in enumerate(strategic_metrics):
        is_ok = mse_model_strat[i].item() < mse_baseline_strat[i].item()
        if is_ok:
            ok_strat += 1
        r2_val = r2(mse_model_strat[i].item(), mse_baseline_strat[i].item())
        mae_z_val = mae_strat[i].item()
        flag = "OK" if is_ok else "MAL"

        if has_real:
            print(f"  {m:<25s} | {r2_val:5.1f}% | {mae_z_val:7.3f} | {'(índice)':>12s} | {flag}")
        else:
            print(f"  {m:<25s} | {r2_val:5.1f}% | {mae_z_val:7.3f} | {flag}")

    # --- Resumen ---
    total_model_raw = mse_model_ind.mean().item() + mse_model_team.mean().item()
    total_baseline_raw = mse_baseline_ind.mean().item() + mse_baseline_team.mean().item()
    r2_raw = (1 - total_model_raw / total_baseline_raw) * 100

    total_model_strat = mse_model_strat.mean().item()
    total_baseline_strat = mse_baseline_strat.mean().item()
    r2_strat = (1 - total_model_strat / total_baseline_strat) * 100

    print(f"\n  RAW METRICS:       R² = {r2_raw:+.1f}%  ({ok_raw}/{total_raw} métricas OK)")
    print(f"  STRATEGIC TARGETS: R² = {r2_strat:+.1f}%  ({ok_strat}/{len(strategic_metrics)} métricas OK)")
    print()


# =====================================================================
#  HELPER
# =====================================================================
def _unpack_model_inputs(batch, device, exp_mode):
    if exp_mode in ('both', 'emb_only'):
        p = batch['player'].to(device)
        a = batch['allies'].to(device)
        e = batch['enemies'].to(device)
    else:
        p = batch['player_pca'].to(device)
        a = batch['allies_pca'].to(device)
        e = batch['enemies_pca'].to(device)

    if exp_mode == 'emb_only':
        ctx = None
    else:
        ctx = batch['team_context'].to(device)

    return p, a, e, ctx


# =====================================================================
#  TRAINING LOOP
# =====================================================================
def train(exp_mode='pca_only'):
    save_path = f"Models/lol_model_{exp_mode}.pth"

    print()
    print("=" * 60)
    print(f"  LoL Win Condition Coach — Experiment: {exp_mode.upper()}")
    print(f"  Targets: {len(INDIVIDUAL_METRICS)} individual + {len(TEAM_METRICS)} team + {len(STRATEGIC_METRICS)} strategic")
    print("=" * 60)

    # 1. Datasets
    train_dataset = LoLDataset(
        TRAIN_FILE, JSON_FILE, INDIVIDUAL_METRICS, TEAM_METRICS,
        STRATEGIC_METRICS, CONTEXT_JSON_FILE,
        experiment_mode=exp_mode, mode='train')
    test_dataset = LoLDataset(
        TEST_FILE, JSON_FILE, INDIVIDUAL_METRICS, TEAM_METRICS,
        STRATEGIC_METRICS, CONTEXT_JSON_FILE,
        experiment_mode=exp_mode, mode='val')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory,
                             persistent_workers=(num_workers > 0))

    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

    # Pre-calcular avg_std para diagnóstico con unidades reales
    with open(JSON_FILE, 'r') as f:
        stats_json = json.load(f)
    avg_std = compute_avg_std_per_metric(stats_json)

    # 2. Modelo
    model = LoLWinConditionModel(
        num_individual_metrics=len(INDIVIDUAL_METRICS),
        num_team_metrics=len(TEAM_METRICS),
        num_strategic_metrics=len(STRATEGIC_METRICS),
        mode=exp_mode,
        num_champions=1000,
        embedding_dim=EMBEDDING_DIM,
        champ_pca_dim=CONTEXT_DIM,
        team_context_dim=CONTEXT_DIM * 2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {total_params:,}")

    criterion = LoLTripleLoss(
        INDIVIDUAL_METRICS, TEAM_METRICS, STRATEGIC_METRICS,
        device=device, team_discount=0.2, strategic_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print(f"--- INICIANDO ENTRENAMIENTO ({exp_mode}) ---\n")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        # === TRAIN ===
        model.train()
        train_loss_total, train_loss_ind, train_loss_team, train_loss_strat = 0, 0, 0, 0

        for batch in train_loader:
            p_in, a_in, e_in, ctx = _unpack_model_inputs(batch, device, exp_mode)
            r = batch['role'].to(device)
            w = batch['win'].to(device)
            y_ind = batch['target_individual'].to(device)
            y_team = batch['target_team'].to(device)
            y_strat = batch['target_strategic'].to(device)

            optimizer.zero_grad()
            pred_ind, pred_team, pred_strat = model(p_in, a_in, e_in, r, w, team_context=ctx)
            loss, l_ind, l_team, l_strat = criterion(
                pred_ind, pred_team, pred_strat, y_ind, y_team, y_strat)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_loss_ind += l_ind.item()
            train_loss_team += l_team.item()
            train_loss_strat += l_strat.item()

        nb_train = len(train_loader)
        avg_train = train_loss_total / nb_train
        avg_train_ind = train_loss_ind / nb_train
        avg_train_team = train_loss_team / nb_train
        avg_train_strat = train_loss_strat / nb_train

        # === VALIDATION ===
        model.eval()
        val_loss_total, val_loss_ind, val_loss_team, val_loss_strat = 0, 0, 0, 0

        with torch.no_grad():
            for batch in test_loader:
                p_in, a_in, e_in, ctx = _unpack_model_inputs(batch, device, exp_mode)
                r = batch['role'].to(device)
                w = batch['win'].to(device)
                y_ind = batch['target_individual'].to(device)
                y_team = batch['target_team'].to(device)
                y_strat = batch['target_strategic'].to(device)

                pred_ind, pred_team, pred_strat = model(p_in, a_in, e_in, r, w, team_context=ctx)
                loss, l_ind, l_team, l_strat = criterion(
                    pred_ind, pred_team, pred_strat, y_ind, y_team, y_strat)

                val_loss_total += loss.item()
                val_loss_ind += l_ind.item()
                val_loss_team += l_team.item()
                val_loss_strat += l_strat.item()

        nb_val = len(test_loader)
        avg_val = val_loss_total / nb_val
        avg_val_ind = val_loss_ind / nb_val
        avg_val_team = val_loss_team / nb_val
        avg_val_strat = val_loss_strat / nb_val

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val)

        print(f"Epoch {epoch+1:03d} | "
              f"Train: {avg_train:.4f} (ind={avg_train_ind:.4f} team={avg_train_team:.4f} strat={avg_train_strat:.4f}) | "
              f"Val: {avg_val:.4f} (ind={avg_val_ind:.4f} team={avg_val_team:.4f} strat={avg_val_strat:.4f}) | "
              f"LR: {current_lr:.6f}")

        if (epoch + 1) % DIAGNOSTIC_EVERY == 0:
            print_diagnostic(model, test_loader, INDIVIDUAL_METRICS, TEAM_METRICS,
                             STRATEGIC_METRICS, device, exp_mode, avg_std=avg_std)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly Stopping en epoch {epoch+1}. Best Val Loss: {best_val_loss:.4f}")
                break

    # --- DIAGNÓSTICO FINAL ---
    print(f"\n--- DIAGNÓSTICO FINAL ({exp_mode}) ---")
    model.load_state_dict(torch.load(save_path, weights_only=True))
    print_diagnostic(model, test_loader, INDIVIDUAL_METRICS, TEAM_METRICS,
                     STRATEGIC_METRICS, device, exp_mode, avg_std=avg_std)
    print(f"Modelo guardado en {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="pca_only",
                        choices=["both", "emb_only", "pca_only"],
                        help="Experimento: 'both', 'emb_only', 'pca_only'")
    args = parser.parse_args()
    train(exp_mode=args.mode)