import torch
import torch.nn as nn


class LoLTripleLoss(nn.Module):
    """
    Loss triple: individual metrics, team metrics, y strategic targets.
    - Individual: 17 métricas brutas por jugador (Huber, weighted)
    - Team: 3 métricas de equipo (Huber, discounted)
    - Strategic: 5 targets estratégicos (Huber, ponderados con prioridad)
    """
    def __init__(self, individual_metrics, team_metrics, strategic_metrics,
                 device, team_discount=0.2, strategic_weight=1.5):
        super().__init__()
        self.individual_metrics = individual_metrics
        self.team_metrics = team_metrics
        self.strategic_metrics = strategic_metrics
        self.huber = nn.SmoothL1Loss(reduction='none')
        
        # --- Pesos individuales (rango moderado 1.0 - 2.0) ---
        individual_weights_config = {
            'GoldDiff_15': 2.0, 'XpDiff_15': 1.5, 'CsDiff_15': 1.5,
            'GoldDiff_10': 1.3, 'XpDiff_10': 1.2, 'CsDiff_10': 1.2,
            'DmgTotal': 1.2, 'KillParticipation': 1.5,
            'SoloKills': 1.2, 'DmgTurret': 1.2, 'DmgObj': 1.2,
        }
        w_ind = [individual_weights_config.get(m, 1.0) for m in individual_metrics]
        self.register_buffer("w_individual",
                             torch.tensor(w_ind, dtype=torch.float32).to(device))
        
        # --- Pesos team ---
        w_team = [1.0] * len(team_metrics)
        self.register_buffer("w_team",
                             torch.tensor(w_team, dtype=torch.float32).to(device))
        
        # --- Pesos strategic ---
        strategic_weights_config = {
            'LaneDominance': 1.5,       # Core — la línea es lo primero
            'EarlyAggression': 1.2,     # Muy útil para consejo táctico
            'MapPressure': 1.3,         # Diferencia elos altos
            'ObjectiveFocus': 1.5,      # Clave para cerrar partidas
            'ResourceEfficiency': 1.0,  # Contexto, menos accionable
        }
        w_strat = [strategic_weights_config.get(m, 1.0) for m in strategic_metrics]
        self.register_buffer("w_strategic",
                             torch.tensor(w_strat, dtype=torch.float32).to(device))
        
        self.team_discount = team_discount
        self.strategic_weight = strategic_weight

    def forward(self, individual_preds, team_preds, strategic_preds,
                individual_targets, team_targets, strategic_targets):
        """Retorna (loss_total, loss_individual, loss_team, loss_strategic) para logging."""
        # Individual
        loss_ind = self.huber(individual_preds, individual_targets)
        loss_ind_weighted = (loss_ind * self.w_individual).mean()
        
        # Team
        loss_team = self.huber(team_preds, team_targets)
        loss_team_weighted = (loss_team * self.w_team).mean() * self.team_discount
        
        # Strategic
        loss_strat = self.huber(strategic_preds, strategic_targets)
        loss_strat_weighted = (loss_strat * self.w_strategic).mean() * self.strategic_weight
        
        loss_total = loss_ind_weighted + loss_team_weighted + loss_strat_weighted
        
        return loss_total, loss_ind_weighted, loss_team_weighted, loss_strat_weighted
    
    def per_metric_mse(self, individual_preds, team_preds, strategic_preds,
                       individual_targets, team_targets, strategic_targets):
        """Para diagnóstico: MSE por métrica sin ponderar."""
        with torch.no_grad():
            mse_ind = ((individual_preds - individual_targets) ** 2).mean(dim=0)
            mse_team = ((team_preds - team_targets) ** 2).mean(dim=0)
            mse_strat = ((strategic_preds - strategic_targets) ** 2).mean(dim=0)
        return mse_ind, mse_team, mse_strat