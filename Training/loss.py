import torch
import torch.nn as nn

class LoLWeightedLoss(nn.Module):
    def __init__(self, metrics_list, device):
        super().__init__()
        self.metrics = metrics_list
        
        # DEFINIMOS LOS PESOS A MANO (Tu conocimiento de LoL es clave aquí)
        # 1.0 es el peso estándar. 
        # > 1.0 obliga al modelo a priorizar esa métrica.
        # < 1.0 le dice que no importa tanto.
        self.weights_config = {
            # --- CRÍTICO (Peso 5.0) ---
            'GoldDiff_15': 5.0,
            'XpDiff_15': 5.0,
            'CsDiff_15': 4.0,
            
            # --- OBJETIVOS DE EQUIPO (Peso 4.0) ---
            'Team_Dragons': 4.0,
            'Team_Barons': 4.0,
            'Team_Towers': 4.0,
            
            # --- COMBATE (Peso 3.0) ---
            'DmgTotal': 3.0,
            'KillParticipation': 3.5,
            'SoloKills': 3.0,
            
            # --- VISIÓN (Peso 1.5 - Importante pero con alta varianza) ---
            'VisionScore': 1.5,
            'WardsPlaced': 1.0, # Poner muchos wards malos no ayuda
            'ControlWardsPlaced': 1.5,
        }
        
        w_list = []
        for m in self.metrics:
            weight = 1.0
            for key, val in self.weights_config.items():
                if key in m:
                    weight = val
                    break
            w_list.append(weight)

        self.register_buffer("weight_tensor", torch.tensor(w_list, dtype=torch.float32).to(device))

    def forward(self, predictions, targets):
        mse = (predictions - targets) ** 2
        return (mse * self.weight_tensor).mean()