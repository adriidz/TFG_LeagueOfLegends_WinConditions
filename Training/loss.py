import torch
import torch.nn as nn

class LoLWeightedLoss(nn.Module):
    def __init__(self, metrics_list):
        super().__init__()
        self.metrics = metrics_list
        
        # DEFINIMOS LOS PESOS A MANO (Tu conocimiento de LoL es clave aquí)
        # 1.0 es el peso estándar. 
        # > 1.0 obliga al modelo a priorizar esa métrica.
        # < 1.0 le dice que no importa tanto.
        self.weights = {
            # --- ECONOMÍA (Lo más importante) ---
            'GoldDiff_15': 5.0,  # ¡CRÍTICO! Saber quién gana la línea
            'XpDiff_15': 4.0,
            'GoldEarned': 3.0,
            
            # --- DAÑO (Importante) ---
            'DmgTotal': 3.0,
            'DmgTurret': 3.0,  # Splitpush
            
            # --- OBJETIVOS (Win Condition) ---
            'Team_Dragons': 4.0,
            'Team_Barons': 4.0,
            'SoloKills': 2.0,
            
            # --- UTILIDAD (Menos peso, mucho ruido) ---
            'WardsPlaced': 0.5,
            'WardsKilled': 0.5,
            'TotalHeal': 1.0,
        }
        
        w_list = []
        for m in self.metrics:
            weight = 1.0
            for key, val in self.weights.items():
                if key in m:
                    weight = val
                    break
            w_list.append(weight)

        self.register_buffer("weight_tensor", torch.tensor(w_list, dtype=torch.float32))

    def forward(self, predictions, targets):
        mse = (predictions - targets) ** 2
        return (mse * self.weight_tensor).mean()