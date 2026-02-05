import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json
import numpy as np
from model import LoLWinConditionModel 

# --- CONFIGURACIÓN ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
EMBEDDING_DIM = 32

# Archivos generados por robust_baseline_calculator.py
TRAIN_FILE = "Data/train_split.csv"
TEST_FILE = "Data/test_split.csv"
JSON_FILE = "Data/champion_stats_robust.json"

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

# --- BUCLE DE ENTRENAMIENTO ---
def train():
    print("--- PREPARANDO DATOS ---")
    
    # 1. Datasets
    train_dataset = LoLDataset(TRAIN_FILE, JSON_FILE, USEFUL_METRICS)
    test_dataset = LoLDataset(TEST_FILE, JSON_FILE, USEFUL_METRICS)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train: {len(train_dataset)} filas | Test: {len(test_dataset)} filas")
    
    # 2. Modelo
    max_id = 1000 # O calcula el max(Input_Player_ID) real
    model = LoLWinConditionModel(num_champions=max_id, num_metrics=len(USEFUL_METRICS), embedding_dim=EMBEDDING_DIM)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("--- INICIANDO ENTRENAMIENTO ---")
    
    for epoch in range(EPOCHS):
        # A) TRAIN LOOP
        model.train()
        train_loss = 0
        for batch in train_loader:
            # Desempaquetar
            p = batch['player']
            a = batch['allies'] # (batch_size, 4)
            e = batch['enemies'] # (batch_size, 5)
            r = batch['role']
            y = batch['target']
            
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
                p = batch['player']
                a = batch['allies']
                e = batch['enemies']
                r = batch['role']
                y = batch['target']
                
                preds = model(p, a, e, r)
                loss = criterion(preds, y)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    # Guardar
    torch.save(model.state_dict(), "lol_model_robust.pth")
    print("\n[OK] Modelo guardado en lol_model_robust.pth")

if __name__ == "__main__":
    train()