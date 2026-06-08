# Embedding analysis comparison

Each row uses the same metadata and train-derived roam means, but a different embedding JSON.

| Embedding | File | Support-only top-k same archetype | Support-only rank-1 same archetype | Roam Pearson r | Roam Spearman rho |
|---|---|---:|---:|---:|---:|
| shared | `C:\Users\adria\Desktop\TFG\final\models\mlp_embed\embeddings_raw.json` | 0.170 | 0.125 | 0.168 | 0.178 |
| per_role_ally_utility | `C:\Users\adria\Desktop\TFG\final\models\mlp_per_role\embeddings_raw_ally_utility.json` | 0.245 | 0.325 | 0.124 | 0.113 |
