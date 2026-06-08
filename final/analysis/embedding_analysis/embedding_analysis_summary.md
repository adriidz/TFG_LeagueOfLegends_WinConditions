# Embedding analysis summary

## Dataset

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

### Neighbors among all champions (Unrestricted)
- Top-5 neighbors that are support-labeled: 0.220
- Top-5 neighbors with same support archetype: 0.040
- Top-5 neighbors with same primary class: 0.155
- Rank-1 nearest neighbor support-labeled rate: 0.300
- Rank-1 same support archetype rate: 0.025

### Neighbors restricted to support champions
- Top-5 support neighbors with same support archetype: 0.170
- Top-5 support neighbors with same primary class: 0.335
- Rank-1 nearest support neighbor with same support archetype rate: 0.125

Rank-1 nearest neighbors for support champions (Unrestricted):

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

## Figure outputs

- `tsne_by_archetype.png`: t-SNE, perplexity=30, support champions colored by support archetype.
- `tsne_perplexity15_by_archetype.png`: t-SNE sensitivity check, perplexity=15.
- `tsne_by_roam_score.png`: t-SNE colored by empirical mean support roam score.
- `embedding_distance_vs_roam.png`: pairwise embedding distance vs. pairwise mean roam-score difference.
- `umap_by_class.png`: UMAP, n_neighbors=15, all champions colored by primary Data Dragon class.
- `umap_n10_by_class.png`: UMAP sensitivity check, n_neighbors=10.
- `support_nearest_neighbors.png`: nearest-neighbor matrix for support champions.

## Suggested wording for the report

The learned embedding space does not align cleanly with discrete human archetypes, but pairwise distances do correlate with differences in observed roaming tendency. This indicates that the MLP embeddings encode a continuous roaming-related signal rather than the expert-defined categories. Architecturally, this signal is diluted by the shared embedding table: the same champion vector is reused across all ten draft slots, so the representation cannot specialize only for ally support.
