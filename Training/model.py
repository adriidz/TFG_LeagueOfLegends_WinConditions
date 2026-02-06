import torch
import torch.nn as nn
import torch.nn.functional as F


class LoLWinConditionModel(nn.Module):
    """
    Multi-head model:
      - individual_head: 17 métricas brutas por jugador
      - team_head: 3 métricas de equipo
      - strategic_head: 5 targets estratégicos (LaneDominance, etc.)
    
    Modos: 'both' | 'emb_only' | 'pca_only'
    """
    def __init__(self, num_individual_metrics, num_team_metrics, num_strategic_metrics=5,
                 mode='pca_only',
                 num_champions=1000, embedding_dim=32,
                 champ_pca_dim=6,
                 team_context_dim=12):
        super().__init__()

        self.mode = mode
        self.num_individual_metrics = num_individual_metrics
        self.num_team_metrics = num_team_metrics
        self.num_strategic_metrics = num_strategic_metrics

        # --- Rol embedding (común) ---
        self.role_embedding = nn.Embedding(5, 8)

        # --- Input dim según modo ---
        if mode == 'both':
            self.champ_embedding = nn.Embedding(num_champions, embedding_dim)
            input_dim = (10 * embedding_dim) + 8 + team_context_dim + 1

        elif mode == 'emb_only':
            self.champ_embedding = nn.Embedding(num_champions, embedding_dim)
            input_dim = (10 * embedding_dim) + 8 + 1

        elif mode == 'pca_only':
            self.champ_pca_dim = champ_pca_dim
            input_dim = (10 * champ_pca_dim) + 8 + team_context_dim + 1
        else:
            raise ValueError(f"mode debe ser 'both', 'emb_only' o 'pca_only', no '{mode}'")

        # --- Backbone compartido ---
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.3)

        # --- Heads ---
        self.individual_head = nn.Linear(128, num_individual_metrics)
        self.team_head = nn.Linear(128, num_team_metrics)
        # Head estratégico con capa intermedia propia (mayor capacidad para abstraer)
        self.strategic_fc = nn.Linear(128, 64)
        self.strategic_bn = nn.BatchNorm1d(64)
        self.strategic_drop = nn.Dropout(0.2)
        self.strategic_head = nn.Linear(64, num_strategic_metrics)

    def forward(self, player_input, allies_input, enemies_input,
                role_id, win_flag, team_context=None):
        r_emb = self.role_embedding(role_id)

        if self.mode in ('both', 'emb_only'):
            p_flat = self.champ_embedding(player_input).view(player_input.size(0), -1)
            a_flat = self.champ_embedding(allies_input).view(allies_input.size(0), -1)
            e_flat = self.champ_embedding(enemies_input).view(enemies_input.size(0), -1)
        else:  # pca_only
            p_flat = player_input
            a_flat = allies_input.view(allies_input.size(0), -1)
            e_flat = enemies_input.view(enemies_input.size(0), -1)

        if self.mode == 'emb_only':
            x = torch.cat([p_flat, a_flat, e_flat, r_emb, win_flag], dim=1)
        else:
            x = torch.cat([p_flat, a_flat, e_flat, r_emb, team_context, win_flag], dim=1)

        # Backbone con skip connection
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)

        identity = x
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = x + identity

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.drop3(x)

        # 3 heads
        out_ind = self.individual_head(x)
        out_team = self.team_head(x)
        
        # Strategic head con capa intermedia
        s = F.relu(self.strategic_bn(self.strategic_fc(x)))
        s = self.strategic_drop(s)
        out_strat = self.strategic_head(s)

        return out_ind, out_team, out_strat