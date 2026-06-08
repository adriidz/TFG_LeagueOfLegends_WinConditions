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

- Support archetype silhouette, cosine metric: -0.131 (n=39, groups=6)
- Primary class silhouette, cosine metric: -0.079 (n=172, groups=7)

Interpretation guide: silhouette near 1 implies compact, separated clusters; near 0 implies overlapping groups; negative values imply that many samples are closer to other groups than to their own.

## Roam-score continuity

- Pairwise support-champion embedding distance vs. absolute mean roam-score difference: Pearson r=0.123, p=5.53e-04; Spearman rho=0.112, p=1.68e-03.
- Pairs analyzed: 780 from 40 support champions.

## Nearest-neighbor structure

- Top-5 neighbors that are support-labeled: 0.295
- Top-5 neighbors with same support archetype: 0.110
- Top-5 neighbors with same primary class: 0.180
- Rank-1 nearest neighbor support-labeled rate: 0.450
- Rank-1 same support archetype rate: 0.125

Rank-1 nearest neighbors for support champions:

| Support | Archetype | Nearest neighbor | Neighbor archetype | Neighbor class | Cosine sim. |
|---|---|---|---|---|---:|
| Morgana | catcher | Viego | - | Assassin | 0.631 |
| Ivern | enchanter | Neeko | mage_support | Mage | 0.634 |
| Janna | enchanter | Quinn | - | Marksman | 0.660 |
| Karma | enchanter | Kayle | enchanter | Fighter | 0.592 |
| Kayle | enchanter | Nami | enchanter | Support | 0.615 |
| Lulu | enchanter | Olaf | - | Fighter | 0.550 |
| Milio | enchanter | Darius | - | Fighter | 0.710 |
| Nami | enchanter | Kayle | enchanter | Fighter | 0.615 |
| Renata Glasc | enchanter | Alistar | engage_tank | Tank | 0.627 |
| Senna | enchanter | Sion | - | Tank | 0.808 |
| Seraphine | enchanter | Garen | - | Fighter | 0.749 |
| Sona | enchanter | Taric | warden | Support | 0.625 |
| Soraka | enchanter | Varus | - | Marksman | 0.600 |
| Yuumi | enchanter | Xerath | mage_support | Mage | 0.601 |
| Zilean | enchanter | Tryndamere | - | Fighter | 0.672 |
| Alistar | engage_tank | Renata Glasc | enchanter | Support | 0.627 |
| Amumu | engage_tank | Poppy | - | Tank | 0.568 |
| Leona | engage_tank | Nautilus | engage_tank | Tank | 0.629 |
| Maokai | engage_tank | Warwick | - | Fighter | 0.648 |
| Nautilus | engage_tank | Leona | engage_tank | Tank | 0.629 |
| Pantheon | engage_tank | Neeko | mage_support | Mage | 0.608 |
| Rakan | engage_tank | Rumble | - | Fighter | 0.481 |
| Rell | engage_tank | Tahm Kench | warden | Support | 0.568 |
| Blitzcrank | hook | Camille | - | Fighter | 0.487 |
| Thresh | hook | Vladimir | - | Mage | 0.560 |
| Annie | mage_support | Karthus | - | Mage | 0.621 |
| Brand | mage_support | Quinn | - | Marksman | 0.631 |
| Fiddlesticks | mage_support | Ashe | - | Marksman | 0.710 |
| Heimerdinger | mage_support | Kayn | - | Fighter | 0.673 |
| Lux | mage_support | Urgot | - | Fighter | 0.650 |
| Neeko | mage_support | Ivern | enchanter | Support | 0.634 |
| Swain | mage_support | Morgana | catcher | Mage | 0.572 |
| Vel'Koz | mage_support | Kalista | - | Marksman | 0.778 |
| Xerath | mage_support | Yuumi | enchanter | Support | 0.601 |
| Zyra | mage_support | Varus | - | Marksman | 0.618 |
| Bard | roamer | Ornn | - | Tank | 0.649 |
| Pyke | roamer | Talon | - | Assassin | 0.554 |
| Braum | warden | Nami | enchanter | Support | 0.546 |
| Tahm Kench | warden | Rell | engage_tank | Tank | 0.568 |
| Taric | warden | Sona | enchanter | Support | 0.625 |

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
