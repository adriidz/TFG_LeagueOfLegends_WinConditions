import torch
import torch.nn as nn
import torch.nn.functional as F

class LoLWinConditionModel(nn.Module):
    def __init__(self, num_champions, num_metrics, embedding_dim=32, context_dim=10):
        """
        Args:
            num_champions: ID máximo de campeón.
            num_metrics: Número de salidas (targets).
            embedding_dim: Tamaño del vector de campeón.
            context_dim: Tamaño del vector de contexto extra (PCA). 
                         Si viene del train.py corregido, será 10 (5 aliados + 5 enemigos).

        """
        super().__init__()
        
        # --- 1. CAPA DE EMBEDDING (La traducción) ---
        # Convierte ID 104 en un vector de 32 números con significado.
        self.champ_embedding = nn.Embedding(num_champions, embedding_dim)
        # También un pequeño embedding para el Rol (0-4)
        self.role_embedding = nn.Embedding(5, 8) 

        # Calculamos el tamaño total que entra a las neuronas
        # (1 Player + 4 Aliados + 5 Enemigos) * 32 dim + 8 dim del rol + contexto PCA
        input_dim = (10 * embedding_dim) + 8 + context_dim

        # --- 2. EL CEREBRO (Basado en la lógica 'Seq_NN' de tu profesora) ---
        # Bloque 1: De Input a 256 neuronas
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Estabiliza el aprendizaje
        self.drop1 = nn.Dropout(0.3)    # Evita memorizar

        # Bloque 2: De 256 a 128 neuronas
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)

        # Bloque 3: De 128 a 64 neuronas
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        # --- 3. SALIDA (Regression Head) ---
        # Predice los Z-Scores. Sin activación final (lineal).
        self.output = nn.Linear(64, num_metrics)

    def forward(self, player_id, ally_ids, enemy_ids, role_id, context_vec):
        """
        player_id: [Batch]
        ally_ids:  [Batch, 4]
        enemy_ids: [Batch, 5]
        role_id:   [Batch]
        context_vec: [Batch, context_dim]
        """
        # A. PROCESAR EMBEDDINGS
        # El jugador principal
        p_emb = self.champ_embedding(player_id) # Shape: [Batch, 32]
        # Aliados y Enemigos
        a_emb = self.champ_embedding(ally_ids)  # Shape: [Batch, 4, 32]
        e_emb = self.champ_embedding(enemy_ids) # Shape: [Batch, 5, 32]
        # El Rol
        r_emb = self.role_embedding(role_id)    # Shape: [Batch, 8]

        # B. APLANAR (Flatten)
        # Convertimos [Batch, 5, 32] a [Batch, 160] para poder juntarlos
        p_flat = p_emb.view(p_emb.size(0), -1) # [Batch, 32]
        a_flat = a_emb.view(a_emb.size(0), -1) # [Batch, 128]
        e_flat = e_emb.view(e_emb.size(0), -1) # [Batch, 160]
        r_flat = r_emb.view(r_emb.size(0), -1) # [Batch, 8]

        # C. CONCATENAR (Juntar todo en una sola fila larga)
        # Esto es lo que entra al cerebro
        x = torch.cat([p_flat, a_flat, e_flat, r_flat, context_vec], dim=1) # Shape: [Batch, 32 + 128 + 160 + 8 + context_dim]

        # D. PASADA POR LAS CAPAS DENSAS (MLP)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        
        # E. PREDICCIÓN FINAL
        out = self.output(x)
        
        return out