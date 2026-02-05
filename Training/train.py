import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import os

# --- IMPORTAMOS TUS ARCHIVOS ---
# Asegúrate de que model.py y el dataset están en la misma carpeta
from model import LoLWinConditionModel 

# Definimos el Dataset aquí mismo para tenerlo todo junto (o impórtalo si lo guardaste aparte)
from torch.utils.data import Dataset
import pandas as pd

# --- CONFIGURACIÓN ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20         # Número de veces que revisa todas las partidas
EMBEDDING_DIM = 32
CSV_FILE = "Data/dataset_train_2.csv"
JSON_FILE = "Data/champion_stats_3.json"

# Lista de métricas (Debe coincidir con la que usaste para filtrar el CSV/JSON)
# COPIA AQUÍ TU LISTA 'USEFUL_METRICS' DEL PASO ANTERIOR
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
    'TotalCS', 'GoldEarned'
]

# --- CLASE DATASET (La que definimos antes) ---
class LoLDataset(Dataset):
    def __init__(self, csv_file, json_file, useful_metrics):
        self.data = pd.read_csv(csv_file)
        with open(json_file, 'r') as f:
            self.stats = json.load(f)
        self.metrics = useful_metrics
        
        # Mapeo de Roles a IDs (0-4)
        self.role_map = {'TOP': 0, 'JUNGLE': 1, 'MIDDLE': 2, 'BOTTOM': 3, 'UTILITY': 4}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 1. Inputs
        player = int(row['Input_Player_ID'])
        role_str = row['Input_Role']
        role_id = self.role_map.get(role_str, 0) # Default 0 si falla
        
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
        
        # 2. Targets (Z-Scores)
        key = f"{player}_{role_str}"
        targets = []
        champ_stats = self.stats.get(key, None)
        
        for metric in self.metrics:
            col_name = f"Target_{metric}"
            if col_name not in row:
                targets.append(0.0) # Si falta la columna, relleno seguro
                continue
                
            real_val = row[col_name]
            
            if champ_stats and col_name in champ_stats:
                mean = champ_stats[col_name]['mean']
                std = champ_stats[col_name]['std']
                # Z-Score
                z = (real_val - mean) / std
                # Clipping para evitar infinitos
                z = max(-5.0, min(5.0, z))
            else:
                z = 0.0
            
            targets.append(z)

        return {
            'player': torch.tensor(player, dtype=torch.long),
            'allies': torch.tensor(allies, dtype=torch.long),
            'enemies': torch.tensor(enemies, dtype=torch.long),
            'role': torch.tensor(role_id, dtype=torch.long),
            'target': torch.tensor(targets, dtype=torch.float32)
        }

# --- BUCLE DE ENTRENAMIENTO ---
def train():
    print("--- INICIANDO ENTRENAMIENTO ---")
    
    # 1. Cargar Datos
    dataset = LoLDataset(CSV_FILE, JSON_FILE, USEFUL_METRICS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Partidas cargadas: {len(dataset)}")
    print(f"Métricas a predecir: {len(USEFUL_METRICS)}")
    
    # 2. Inicializar Modelo
    # Buscamos el ID más alto para dimensionar los Embeddings
    # (Truco: sumamos 1000 por seguridad)
    max_id = 1000 
    
    model = LoLWinConditionModel(
        num_champions=max_id, 
        num_metrics=len(USEFUL_METRICS),
        embedding_dim=EMBEDDING_DIM
    )
    
    # 3. Optimizador y Función de Pérdida
    criterion = nn.MSELoss() # Error Cuadrático Medio (Estándar para regresión)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for batch in dataloader:
            # Desempaquetar batch
            p_id = batch['player']
            a_ids = batch['allies']
            e_ids = batch['enemies']
            r_id = batch['role']
            targets = batch['target']
            
            # Forward (Predicción)
            preds = model(p_id, a_ids, e_ids, r_id)
            
            # Loss (Error)
            loss = criterion(preds, targets)
            
            # Backward (Aprender)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
        
    # 5. Guardar el Modelo Entrenado
    torch.save(model.state_dict(), "lol_model.pth")
    print("\n--- ENTRENAMIENTO FINALIZADO ---")
    print("Modelo guardado en: lol_model.pth")

if __name__ == "__main__":
    train()