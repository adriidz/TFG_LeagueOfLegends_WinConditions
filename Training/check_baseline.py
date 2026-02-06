import torch
from loss import LoLWeightedLoss
from train import USEFUL_METRICS

# Simular predicciones "Promedio" (todo ceros)
# Batch size 1, num_metrics filas
dummy_preds = torch.zeros(1, len(USEFUL_METRICS))

# Simular un Target real típico (digamos, Z-Score de 1.0 en todo para probar)
# O mejor, usa un batch real de tu dataloader si puedes.
# Para una estimación teórica: Si asumimos que los targets reales son N(0, 1)
# El MSE esperado de predecir 0 es 1.0 * Peso.
criterion = LoLWeightedLoss(USEFUL_METRICS, device='cpu')

print("Pesos:", criterion.weight_tensor)
print("Baseline Loss Teórica (si el modelo predice 0 y el target es 1):", criterion.weight_tensor.mean().item())