import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json
import numpy as np
import os
from model import LoLWinConditionModel 
from loss import LoLWeightedLoss

# --- CONFIGURACIÓN ---
BATCH_SIZE = 256
LEARNING_RATE = 0.0005
EPOCHS = 100
EMBEDDING_DIM = 32
CONTEXT_DIM = 6

EARLY_STOPPING_PATIENCE = 10   # epochs sin mejorar antes de parar
SAVE_BEST_PATH = "Models/lol_model_best.pth"

# Archivos generados por robust_baseline_calculator.py
TRAIN_FILE = "Data/train_split.csv"
TEST_FILE = "Data/test_split.csv"
JSON_FILE = "Data/champion_stats_3.json"
CONTEXT_JSON_FILE = "Data/champion_context_pca.json"

# TU LISTA DE MÉTRICAS (Asegúrate de que es la misma que usaste antes)
SAVED_METRICS = [
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

USEFUL_METRICS = [
    'GoldDiff_10', 'XpDiff_10', 'CsDiff_10',
    'GoldDiff_15', 'XpDiff_15', 'CsDiff_15',
    'EarlyTakedowns', 'LaneCsBefore10', 'JgCsBefore10',
    'DmgTotal', 'DmgTurret', 'DmgObj', 'VisionScore',
    'KillParticipation', 'SoloKills', 
    'TotalCS', 'GoldEarned',
    'Team_Dragons', 'Team_Barons', 'Team_Towers'
]

# --- CLASE DATASET ---
class LoLDataset(Dataset):
    def __init__(self, csv_file, json_file, useful_metrics, context_json, mode='train'):
        """
        mode: 'train' o 'val'. 
        """
        # 1. Cargamos CSV crudo
        df_raw = pd.read_csv(csv_file)

        # 2. FILTRO DE BEHAVIORAL CLONING
        # Queremos aprender SOLO de los que ganaron.
        # El JSON de stats (json_file) ya tiene la media global (Wins+Losses),
        # así que compararemos a los ganadores contra el promedio global.
        self.data = df_raw[df_raw['win'] == 1].copy().reset_index(drop=True)

        print(f"Dataset ({mode}): Cargadas {len(self.data)} partidas ganadoras (de un total de {len(df_raw)}).")

        # 3. Cargar Stats Globales (Norma)
        with open(json_file, 'r') as f:
            self.stats = json.load(f)
        
        self.metrics = useful_metrics
        self.role_map = {'TOP': 0, 'JUNGLE': 1, 'MIDDLE': 2, 'BOTTOM': 3, 'UTILITY': 4}

        # Cargamos el contexto PCA pre-calculado
        with open(context_json, 'r') as f:
            self.pca_context = json.load(f)
            
        # Vector de ceros por si falta algún campeón (fallback)
        self.context_dim = 5 # Debe coincidir con N_COMPONENTS
        self.empty_context = [0.0] * self.context_dim

    def get_team_context_pca(self, team_tuples):
        """
        Suma los vectores PCA de los 5 campeones del equipo.
        team_tuples: Lista de [(ID, 'TOP'), (ID, 'JUNGLE')...]
        """
        # Inicializamos vector de ceros
        team_vector = np.zeros(self.context_dim, dtype=np.float32)
        
        for cid, role in team_tuples:
            key = f"{cid}_{role}"
            # Buscamos en el JSON PCA, si no existe, devolvemos ceros
            vec = self.pca_context.get(key, self.empty_context)
            team_vector += np.array(vec, dtype=np.float32)
            
        return team_vector

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Inputs
        player = int(row['Input_Player_ID'])
        role_str = row['Input_Role']
        role_id = self.role_map.get(role_str, 0)

        # --- 2. RECONSTRUCCIÓN DE EQUIPOS (Para el Contexto) ---
        roles_order = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
        allies_tuples = []
        enemies_tuples = []
        
        # Inputs normales (IDs)
        allies_ids_input = []
        enemies_ids_input = []
        
        for r in roles_order:
            a_id = int(row[f'Input_Ally_{r}_ID'])
            e_id = int(row[f'Input_Enemy_{r}_ID'])
            
            # Guardamos tuplas para buscar en el JSON PCA
            allies_tuples.append((a_id, r))
            enemies_tuples.append((e_id, r))

            # Guardamos IDs para el modelo (excluyendo al jugador principal en aliados)
            if r != role_str:
                allies_ids_input.append(a_id)
            enemies_ids_input.append(e_id)

        # --- 3. CÁLCULO DE CONTEXTO ---
        # Vector Aliado (5 dimensiones) + Vector Enemigo (5 dimensiones)
        ctx_ally = self.get_team_context_pca(allies_tuples)
        ctx_enemy = self.get_team_context_pca(enemies_tuples)

        context_vector = np.concatenate([ctx_ally, ctx_enemy]) # Total 10 dimensiones de contexto

        # --- 4. TARGETS (Z-Scores) ---
        key = f"{player}_{role_str}"
        champ_stats = self.stats.get(key, {})
        z_scores = []

        for metric in self.metrics:
            col_name = f"Target_{metric}"
            real_val = row.get(col_name, 0.0)

            if col_name in champ_stats:
                mean_global = champ_stats[col_name]["mean"]
                std_global = champ_stats[col_name]["std"]
                z = (real_val - mean_global) / std_global if std_global > 0 else 0.0
                z = max(-4.0, min(4.0, z)) # Clipping para evitar outliers extremos
            else:
                z = 0.0 # Si no hay datos, asumimos promedio (0)
            z_scores.append(float(z))

        return {
            'player': torch.tensor(player, dtype=torch.long),
            'allies': torch.tensor(allies_ids_input, dtype=torch.long), # Solo los 4 teammates restantes
            'enemies': torch.tensor(enemies_ids_input, dtype=torch.long), # Los 5 enemigos
            'role': torch.tensor(role_id, dtype=torch.long),
            'context': torch.tensor(context_vector, dtype=torch.float32),
            'target': torch.tensor(z_scores, dtype=torch.float32)
        }

# --- BUCLE DE ENTRENAMIENTO ---
def train():
    print()
    print("--- PREPARANDO DATOS ---")
    
    # 1. Datasets
    train_dataset = LoLDataset(TRAIN_FILE, JSON_FILE, USEFUL_METRICS, context_json=CONTEXT_JSON_FILE, mode='train')
    test_dataset = LoLDataset(TEST_FILE, JSON_FILE, USEFUL_METRICS, context_json=CONTEXT_JSON_FILE, mode='val')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    print(f"Usando num_workers={num_workers} para DataLoader (SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK', 'N/A')})")
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=(num_workers > 0))
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers,
                            persistent_workers=(num_workers > 0))
    
    print(f"Train: {len(train_dataset)} filas | Test: {len(test_dataset)} filas")
    
    # 2. Modelo
    max_id = 1000 # O calcula el max(Input_Player_ID) real
    model = LoLWinConditionModel(num_champions=max_id,
                                 num_metrics=len(USEFUL_METRICS),
                                 embedding_dim=EMBEDDING_DIM,
                                 context_dim=CONTEXT_DIM*2).to(device)
    
    criterion = LoLWeightedLoss(USEFUL_METRICS, device=device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    
    print("--- INICIANDO ENTRENAMIENTO ---")
    
    best_loss = float("inf")
    patience_counter = 0

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
            c = batch['context'].to(device) # (batch_size, 10)
            y = batch['target'].to(device) # (batch_size, num_metrics)
            
            optimizer.zero_grad()

            preds = model(p, a, e, r, c)

            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # B) VALIDATION LOOP (¡Sin Data Leakage!)
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                p = batch['player'].to(device)
                a = batch['allies'].to(device)
                e = batch['enemies'].to(device)
                r = batch['role'].to(device)
                c = batch['context'].to(device)
                y = batch['target'].to(device)  # [B, 8]

                preds = model(p, a, e, r, c)       # [B, 8]
                loss = criterion(preds, y) # Usamos la misma Loss ponderada para validar
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_BEST_PATH)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early Stopping activado.")
                break

    print(f"Entrenamiento finalizado. Modelo guardado en {SAVE_BEST_PATH}")

if __name__ == "__main__":
    train()