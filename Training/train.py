import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json
import numpy as np
import os
from model import LoLWinConditionModel 

# --- CONFIGURACIÓN ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 200
EMBEDDING_DIM = 32

EARLY_STOPPING_PATIENCE = 20   # epochs sin mejorar antes de parar
EARLY_STOPPING_MIN_DELTA = 1e-4
SAVE_BEST_PATH = "Models/lol_model_best.pth"

# Archivos generados por robust_baseline_calculator.py
TRAIN_FILE = "Data/train_split.csv"
TEST_FILE = "Data/test_split.csv"
JSON_FILE = "Data/champion_stats_3.json"

# TU LISTA DE MÉTRICAS (Asegúrate de que es la misma que usaste antes)
USEFUL_METRICS = [
    'GoldDiff_5', 'XpDiff_5', 'CsDiff_5', 
    'GoldDiff_10', 'XpDiff_10', 'CsDiff_10',
    'GoldDiff_15', 'XpDiff_15', 'CsDiff_15',
    'GoldDiff_20', 'XpDiff_20', 'CsDiff_20',
    'EarlyTakedowns', 'LaneCsBefore10', 'JgCsBefore10', 'EnemyJgInvades',
    'DmgTotal', 'DmgPhys', 'DmgMagic', 'DmgTrue', 'TimeCC',
    'TotalHeal', 'HealOnTeammates', 'DamageMitigated', 'KillParticipation', 'SoloKills',
    'DmgTurret', 'TurretPlates', 'DmgObj', 
    'DragonTakedowns', 'BaronTakedowns', 'RiftHeraldTakedowns', 'VoidMonsterTakedowns',
    'VisionScore', 'WardsPlaced', 'WardsKilled', 'ControlWardsPlaced',
    'TotalCS', 'GoldEarned',
    'Team_Dragons', 'Team_Barons', 'Team_Towers', 'Team_Inhibitors' # Agregamos objetivos de equipo si están
]

# --- CLASE DATASET ---
class LoLDataset(Dataset):
    def __init__(self, csv_file, json_file, useful_metrics):
        self.data = pd.read_csv(csv_file)
        with open(json_file, 'r') as f:
            self.stats = json.load(f)
        self.metrics = useful_metrics
        self.role_map = {'TOP': 0, 'JUNGLE': 1, 'MIDDLE': 2, 'BOTTOM': 3, 'UTILITY': 4}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Inputs
        player = int(row['Input_Player_ID'])
        role_str = row['Input_Role']
        role_id = self.role_map.get(role_str, 0)
        
        allies = [
            int(row['Input_Ally_TOP_ID']), int(row['Input_Ally_JUNGLE_ID']),
            int(row['Input_Ally_MIDDLE_ID']), int(row['Input_Ally_BOTTOM_ID']),
            int(row['Input_Ally_UTILITY_ID'])
        ]
        
        enemies = [
            int(row['Input_Enemy_TOP_ID']), int(row['Input_Enemy_JUNGLE_ID']),
            int(row['Input_Enemy_MIDDLE_ID']), int(row['Input_Enemy_BOTTOM_ID']),
            int(row['Input_Enemy_UTILITY_ID'])
        ]

        # El jugador principal (player) ya está en allies[role_id], lo eliminamos para quedarnos con 4 teammates
        allies_other = allies.copy()
        allies_other.pop(role_id)  # len = 4
        
        # Targets (Z-Scores)
        key = f"{player}_{role_str}"
        targets = []
        champ_stats = self.stats.get(key, None)
        
        for metric in self.metrics:
            col_name = f"Target_{metric}"
            
            # Si no existe la columna en el CSV o no hay stats en el JSON
            if col_name not in row:
                targets.append(0.0)
                continue
            
            real_val = row[col_name]
            
            if champ_stats and col_name in champ_stats:
                mean = champ_stats[col_name]['mean']
                std = champ_stats[col_name]['std']
                z = (real_val - mean) / std
                z = max(-5.0, min(5.0, z)) # Clip
            else:
                z = 0.0 # Fallback final
            
            targets.append(z)

        return {
            'player': torch.tensor(player, dtype=torch.long),
            'allies': torch.tensor(allies_other, dtype=torch.long), # Solo los 4 teammates restantes
            'enemies': torch.tensor(enemies, dtype=torch.long), # Los 5 enemigos
            'role': torch.tensor(role_id, dtype=torch.long),
            'target': torch.tensor(targets, dtype=torch.float32)
        }

def evaluate_model_and_baseline(model, data_loader, device, num_metrics):
    """
    Devuelve:
      - model_mse_global: float
      - baseline_mse_global: float  (baseline = predecir 0)
      - model_mse_per_metric: np.ndarray shape [num_metrics]
      - baseline_mse_per_metric: np.ndarray shape [num_metrics]
    """
    model.eval()

    sum_sqerr_model = torch.zeros(num_metrics, device=device)
    sum_sqerr_base = torch.zeros(num_metrics, device=device)
    n_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            p = batch['player'].to(device)
            a = batch['allies'].to(device)
            e = batch['enemies'].to(device)
            r = batch['role'].to(device)
            y = batch['target'].to(device)  # [B, M]

            preds = model(p, a, e, r)       # [B, M]

            # Modelo
            sqerr_model = (preds - y) ** 2  # [B, M]
            sum_sqerr_model += sqerr_model.sum(dim=0)

            # Baseline: predecir 0
            sqerr_base = (0.0 - y) ** 2
            sum_sqerr_base += sqerr_base.sum(dim=0)

            n_samples += y.size(0)

    model_mse_per_metric = (sum_sqerr_model / n_samples).detach().cpu().numpy()
    base_mse_per_metric = (sum_sqerr_base / n_samples).detach().cpu().numpy()

    model_mse_global = float(model_mse_per_metric.mean())
    base_mse_global = float(base_mse_per_metric.mean())
    
    return model_mse_global, base_mse_global, model_mse_per_metric, base_mse_per_metric

def print_metric_report(metric_names, model_mse_per_metric, base_mse_per_metric, top_k=10):
    rows = []
    for name, m_mse, b_mse in zip(metric_names, model_mse_per_metric, base_mse_per_metric):
        rows.append((name, float(m_mse), float(b_mse), float(b_mse - m_mse)))

    # Ordenar por “mejora sobre baseline” (más positivo = mejor)
    rows.sort(key=lambda x: x[3], reverse=True)

    print("\n--- BASELINE vs MODELO (TEST) ---")
    print(f"Mejoras (baseline_mse - model_mse), top {top_k}:")
    for name, m_mse, b_mse, gain in rows[:top_k]:
        print(f"  {name:24s} | model={m_mse:.4f} | base0={b_mse:.4f} | gain={gain:+.4f}")

    print(f"\nPeores (el modelo empeora), top {top_k}:")
    for name, m_mse, b_mse, gain in rows[-top_k:]:
        print(f"  {name:24s} | model={m_mse:.4f} | base0={b_mse:.4f} | gain={gain:+.4f}")

# --- BUCLE DE ENTRENAMIENTO ---
def train():
    print()
    print("--- PREPARANDO DATOS ---")
    
    # 1. Datasets
    train_dataset = LoLDataset(TRAIN_FILE, JSON_FILE, USEFUL_METRICS)
    test_dataset = LoLDataset(TEST_FILE, JSON_FILE, USEFUL_METRICS)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=(num_workers > 0))
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=(num_workers > 0))
    
    print(f"Train: {len(train_dataset)} filas | Test: {len(test_dataset)} filas")
    
    # 2. Modelo
    max_id = 1000 # O calcula el max(Input_Player_ID) real
    model = LoLWinConditionModel(num_champions=max_id, num_metrics=len(USEFUL_METRICS), embedding_dim=EMBEDDING_DIM).to(device)
    
    print("\n--- EVALUACIÓN INICIAL (sin entrenar) ---")
    m_global, b_global, m_per, b_per = evaluate_model_and_baseline(
        model, test_loader, device, num_metrics=len(USEFUL_METRICS)
    )
    print(f"Global MSE modelo (init): {m_global:.4f}")
    print(f"Global MSE baseline(0):   {b_global:.4f}")
    print()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("--- INICIANDO ENTRENAMIENTO ---")
    
    best_test_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        # A) TRAIN LOOP
        model.train()
        train_loss = 0
        for batch in train_loader:
            # Desempaquetar
            p = batch['player'].to(device) # (batch_size,)
            a = batch['allies'].to(device) # (batch_size, 4)
            e = batch['enemies'].to(device) # (batch_size, 5)
            r = batch['role'].to(device) # (batch_size,)
            y = batch['target'].to(device) # (batch_size, num_metrics)
            
            optimizer.zero_grad()
            preds = model(p, a, e, r)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # B) VALIDATION LOOP (¡Sin Data Leakage!)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                p = batch['player'].to(device) # (batch_size,)
                a = batch['allies'].to(device) # (batch_size, 4)
                e = batch['enemies'].to(device) # (batch_size, 5)
                r = batch['role'].to(device) # (batch_size,)
                y = batch['target'].to(device) # (batch_size, num_metrics)
                
                preds = model(p, a, e, r)
                loss = criterion(preds, y)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)

        improved = avg_test_loss < (best_test_loss - EARLY_STOPPING_MIN_DELTA)

        epoch_message = f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}"

        if improved:
            best_test_loss = avg_test_loss
            epochs_no_improve = 0
            
            # Guardar “best” en CPU sin mover el modelo entero
            state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(state_cpu, SAVE_BEST_PATH)
            print(f"{epoch_message} || [OK] Best model guardado: {SAVE_BEST_PATH} (loss={best_test_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"{epoch_message} || [EarlyStopping] no mejora: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\n[EarlyStopping] Deteniendo entrenamiento después de {epoch+1} epochs sin mejora.")
                break

    print("\n--- EVALUACIÓN FINAL (best checkpoint) ---")
    if os.path.exists(SAVE_BEST_PATH):
        model.load_state_dict(torch.load(SAVE_BEST_PATH, map_location=device))

    m_global, b_global, m_per, b_per = evaluate_model_and_baseline(
        model, test_loader, device, num_metrics=len(USEFUL_METRICS)
    )
    print(f"Global MSE modelo (best): {m_global:.4f}")
    print(f"Global MSE baseline(0):   {b_global:.4f}")
    print_metric_report(USEFUL_METRICS, m_per, b_per, top_k=10)

    print(f"\n[OK] Best checkpoint ya guardado en {SAVE_BEST_PATH}")

if __name__ == "__main__":
    train()