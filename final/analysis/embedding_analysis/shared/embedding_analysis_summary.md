# Embedding analysis summary

## Dataset

- Embedding label: `shared`
- Embedding file analyzed: `C:\Users\adria\Desktop\TFG\final\models\mlp_embed\embeddings_raw.json`
- Champions with embeddings: 172
- Support-labeled champions: 40
- Embedding dimension: 16

Support archetype counts:

- catcher: 1
- enchanter: 14
- engage_tank: 8
- hook: 2
- mage_support: 10
- roamer: 2
- warden: 3

Primary Data Dragon class counts:

- Assassin: 19
- Fighter: 46
- Mage: 36
- Marksman: 28
- Support: 17
- Tank: 21
- Unknown: 5

## Cluster coherence

- Support archetype silhouette, cosine metric: -0.145 (n=39, groups=6)
- Primary class silhouette, cosine metric: -0.074 (n=172, groups=7)

Interpretation guide: silhouette near 1 implies compact, separated clusters; near 0 implies overlapping groups; negative values imply that many samples are closer to other groups than to their own.

## Roam-score continuity

- Pairwise support-champion embedding distance vs. absolute mean roam-score difference: Pearson r=0.168, p=2.25e-06; Spearman rho=0.178, p=5.63e-07.
- Pairs analyzed: 780 from 40 support champions.

## Nearest-neighbor structure

### Neighbors among all champions (Exploratory / unrestricted)
These neighbors are useful for inspection, but they can mix support and non-support champions and should not be used as the main support-archetype check.
- Top-5 neighbors that are support-labeled: 0.220
- Top-5 neighbors with same support archetype: 0.040
- Top-5 neighbors with same primary class: 0.155
- Rank-1 nearest neighbor support-labeled rate: 0.300
- Rank-1 same support archetype rate: 0.025

### Neighbors restricted to support champions (Main support-only check)
- Top-5 support neighbors with same support archetype: 0.170
- Top-5 support neighbors with same primary class: 0.335
- Rank-1 nearest support neighbor with same support archetype rate: 0.125

Rank-1 nearest neighbors for support champions (all champions, exploratory):

| Support | Archetype | Nearest neighbor | Neighbor archetype | Neighbor class | Cosine sim. |
|---|---|---|---|---|---:|
| Morgana | catcher | Rakan | engage_tank | Support | 0.741 |
| Ivern | enchanter | Swain | mage_support | Mage | 0.602 |
| Janna | enchanter | Riven | - | Fighter | 0.793 |
| Karma | enchanter | Singed | - | Tank | 0.571 |
| Kayle | enchanter | Kassadin | - | Assassin | 0.602 |
| Lulu | enchanter | Wukong | - | Fighter | 0.608 |
| Milio | enchanter | Kennen | - | Mage | 0.549 |
| Nami | enchanter | Vel'Koz | mage_support | Mage | 0.595 |
| Renata Glasc | enchanter | Sylas | - | Mage | 0.552 |
| Senna | enchanter | Ryze | - | Mage | 0.623 |
| Seraphine | enchanter | Malzahar | - | Mage | 0.581 |
| Sona | enchanter | Lulu | enchanter | Support | 0.531 |
| Soraka | enchanter | Kalista | - | Marksman | 0.556 |
| Yuumi | enchanter | Xerath | mage_support | Mage | 0.631 |
| Zilean | enchanter | champ_800 | - | Unknown | 0.603 |
| Alistar | engage_tank | Vex | - | Mage | 0.688 |
| Amumu | engage_tank | Malzahar | - | Mage | 0.616 |
| Leona | engage_tank | Xin Zhao | - | Fighter | 0.531 |
| Maokai | engage_tank | Lissandra | - | Mage | 0.613 |
| Nautilus | engage_tank | Jax | - | Fighter | 0.589 |
| Pantheon | engage_tank | Ekko | - | Assassin | 0.656 |
| Rakan | engage_tank | Morgana | catcher | Mage | 0.741 |
| Rell | engage_tank | Diana | - | Fighter | 0.662 |
| Blitzcrank | hook | Riven | - | Fighter | 0.603 |
| Thresh | hook | Heimerdinger | mage_support | Mage | 0.617 |
| Annie | mage_support | Volibear | - | Fighter | 0.624 |
| Brand | mage_support | Talon | - | Assassin | 0.633 |
| Fiddlesticks | mage_support | Janna | enchanter | Support | 0.634 |
| Heimerdinger | mage_support | Thresh | hook | Support | 0.617 |
| Lux | mage_support | Quinn | - | Marksman | 0.571 |
| Neeko | mage_support | Nautilus | engage_tank | Tank | 0.551 |
| Swain | mage_support | Ivern | enchanter | Support | 0.602 |
| Vel'Koz | mage_support | Karthus | - | Mage | 0.631 |
| Xerath | mage_support | Lillia | - | Fighter | 0.682 |
| Zyra | mage_support | Sejuani | - | Tank | 0.652 |
| Bard | roamer | Zac | - | Tank | 0.590 |
| Pyke | roamer | Briar | - | Fighter | 0.658 |
| Braum | warden | Kalista | - | Marksman | 0.606 |
| Tahm Kench | warden | Nilah | - | Fighter | 0.662 |
| Taric | warden | Milio | enchanter | Support | 0.546 |

Rank-1 nearest neighbors restricted to support champions:

| Support | Archetype | Nearest support neighbor | Neighbor archetype | Neighbor class | Cosine sim. |
|---|---|---|---|---|---:|
| Morgana | catcher | Rakan | engage_tank | Support | 0.741 |
| Ivern | enchanter | Swain | mage_support | Mage | 0.602 |
| Janna | enchanter | Fiddlesticks | mage_support | Mage | 0.634 |
| Karma | enchanter | Blitzcrank | hook | Tank | 0.370 |
| Kayle | enchanter | Pantheon | engage_tank | Fighter | 0.435 |
| Lulu | enchanter | Sona | enchanter | Support | 0.531 |
| Milio | enchanter | Taric | warden | Support | 0.546 |
| Nami | enchanter | Vel'Koz | mage_support | Mage | 0.595 |
| Renata Glasc | enchanter | Bard | roamer | Support | 0.415 |
| Senna | enchanter | Annie | mage_support | Mage | 0.347 |
| Seraphine | enchanter | Amumu | engage_tank | Tank | 0.460 |
| Sona | enchanter | Lulu | enchanter | Support | 0.531 |
| Soraka | enchanter | Tahm Kench | warden | Support | 0.355 |
| Yuumi | enchanter | Xerath | mage_support | Mage | 0.631 |
| Zilean | enchanter | Morgana | catcher | Mage | 0.480 |
| Alistar | engage_tank | Annie | mage_support | Mage | 0.617 |
| Amumu | engage_tank | Seraphine | enchanter | Mage | 0.460 |
| Leona | engage_tank | Maokai | engage_tank | Tank | 0.335 |
| Maokai | engage_tank | Nautilus | engage_tank | Tank | 0.557 |
| Nautilus | engage_tank | Maokai | engage_tank | Tank | 0.557 |
| Pantheon | engage_tank | Kayle | enchanter | Fighter | 0.435 |
| Rakan | engage_tank | Morgana | catcher | Mage | 0.741 |
| Rell | engage_tank | Fiddlesticks | mage_support | Mage | 0.423 |
| Blitzcrank | hook | Janna | enchanter | Support | 0.430 |
| Thresh | hook | Heimerdinger | mage_support | Mage | 0.617 |
| Annie | mage_support | Alistar | engage_tank | Tank | 0.617 |
| Brand | mage_support | Maokai | engage_tank | Tank | 0.493 |
| Fiddlesticks | mage_support | Janna | enchanter | Support | 0.634 |
| Heimerdinger | mage_support | Thresh | hook | Support | 0.617 |
| Lux | mage_support | Morgana | catcher | Mage | 0.407 |
| Neeko | mage_support | Nautilus | engage_tank | Tank | 0.551 |
| Swain | mage_support | Ivern | enchanter | Support | 0.602 |
| Vel'Koz | mage_support | Nami | enchanter | Support | 0.595 |
| Xerath | mage_support | Yuumi | enchanter | Support | 0.631 |
| Zyra | mage_support | Yuumi | enchanter | Support | 0.436 |
| Bard | roamer | Tahm Kench | warden | Support | 0.452 |
| Pyke | roamer | Rakan | engage_tank | Support | 0.558 |
| Braum | warden | Lulu | enchanter | Support | 0.388 |
| Tahm Kench | warden | Janna | enchanter | Support | 0.456 |
| Taric | warden | Milio | enchanter | Support | 0.546 |

## Figure outputs

- Projection figures were skipped because `--skip-projections` was used.
- `embedding_distance_vs_roam.png`: pairwise embedding distance vs. pairwise mean roam-score difference.
- `support_only_nearest_neighbors.png`: nearest-neighbor matrix restricted to support champions.

## Suggested wording for the report

The embedding space shows measurable continuity with empirical roaming tendency: support-pair embedding distance increases with pairwise mean roam-score difference. This supports a cautious task-specific interpretation, but it is not by itself evidence of a general semantic champion map.
