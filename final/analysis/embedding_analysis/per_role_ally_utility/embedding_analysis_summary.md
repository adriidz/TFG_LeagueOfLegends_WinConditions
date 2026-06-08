# Embedding analysis summary

## Dataset

- Embedding label: `per_role_ally_utility`
- Embedding file analyzed: `C:\Users\adria\Desktop\TFG\final\models\mlp_per_role\embeddings_raw_ally_utility.json`
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

- Pairwise support-champion embedding distance vs. absolute mean roam-score difference: Pearson r=0.124, p=5.25e-04; Spearman rho=0.113, p=1.59e-03.
- Pairs analyzed: 780 from 40 support champions.

## Nearest-neighbor structure

### Neighbors among all champions (Exploratory / unrestricted)
These neighbors are useful for inspection, but they can mix support and non-support champions and should not be used as the main support-archetype check.
- Top-5 neighbors that are support-labeled: 0.295
- Top-5 neighbors with same support archetype: 0.110
- Top-5 neighbors with same primary class: 0.180
- Rank-1 nearest neighbor support-labeled rate: 0.450
- Rank-1 same support archetype rate: 0.125

### Neighbors restricted to support champions (Main support-only check)
- Top-5 support neighbors with same support archetype: 0.245
- Top-5 support neighbors with same primary class: 0.315
- Rank-1 nearest support neighbor with same support archetype rate: 0.325

Rank-1 nearest neighbors for support champions (all champions, exploratory):

| Support | Archetype | Nearest neighbor | Neighbor archetype | Neighbor class | Cosine sim. |
|---|---|---|---|---|---:|
| Morgana | catcher | Viego | - | Assassin | 0.631 |
| Ivern | enchanter | Neeko | mage_support | Mage | 0.633 |
| Janna | enchanter | Quinn | - | Marksman | 0.661 |
| Karma | enchanter | Kayle | enchanter | Fighter | 0.593 |
| Kayle | enchanter | Nami | enchanter | Support | 0.615 |
| Lulu | enchanter | Olaf | - | Fighter | 0.553 |
| Milio | enchanter | Darius | - | Fighter | 0.712 |
| Nami | enchanter | Kayle | enchanter | Fighter | 0.615 |
| Renata Glasc | enchanter | Alistar | engage_tank | Tank | 0.626 |
| Senna | enchanter | Sion | - | Tank | 0.809 |
| Seraphine | enchanter | Garen | - | Fighter | 0.747 |
| Sona | enchanter | Taric | warden | Support | 0.627 |
| Soraka | enchanter | Varus | - | Marksman | 0.600 |
| Yuumi | enchanter | Xerath | mage_support | Mage | 0.604 |
| Zilean | enchanter | Tryndamere | - | Fighter | 0.672 |
| Alistar | engage_tank | Renata Glasc | enchanter | Support | 0.626 |
| Amumu | engage_tank | Poppy | - | Tank | 0.567 |
| Leona | engage_tank | Nautilus | engage_tank | Tank | 0.630 |
| Maokai | engage_tank | Warwick | - | Fighter | 0.648 |
| Nautilus | engage_tank | Leona | engage_tank | Tank | 0.630 |
| Pantheon | engage_tank | Neeko | mage_support | Mage | 0.608 |
| Rakan | engage_tank | Rumble | - | Fighter | 0.480 |
| Rell | engage_tank | Tahm Kench | warden | Support | 0.567 |
| Blitzcrank | hook | Camille | - | Fighter | 0.486 |
| Thresh | hook | Vladimir | - | Mage | 0.561 |
| Annie | mage_support | Karthus | - | Mage | 0.620 |
| Brand | mage_support | Quinn | - | Marksman | 0.633 |
| Fiddlesticks | mage_support | Ashe | - | Marksman | 0.706 |
| Heimerdinger | mage_support | Kayn | - | Fighter | 0.672 |
| Lux | mage_support | Urgot | - | Fighter | 0.652 |
| Neeko | mage_support | Ivern | enchanter | Support | 0.633 |
| Swain | mage_support | Morgana | catcher | Mage | 0.570 |
| Vel'Koz | mage_support | Kalista | - | Marksman | 0.779 |
| Xerath | mage_support | Yuumi | enchanter | Support | 0.604 |
| Zyra | mage_support | Varus | - | Marksman | 0.618 |
| Bard | roamer | Ornn | - | Tank | 0.650 |
| Pyke | roamer | Talon | - | Assassin | 0.555 |
| Braum | warden | Nami | enchanter | Support | 0.548 |
| Tahm Kench | warden | Rell | engage_tank | Tank | 0.567 |
| Taric | warden | Sona | enchanter | Support | 0.627 |

Rank-1 nearest neighbors restricted to support champions:

| Support | Archetype | Nearest support neighbor | Neighbor archetype | Neighbor class | Cosine sim. |
|---|---|---|---|---|---:|
| Morgana | catcher | Swain | mage_support | Mage | 0.570 |
| Ivern | enchanter | Neeko | mage_support | Mage | 0.633 |
| Janna | enchanter | Ivern | enchanter | Support | 0.573 |
| Karma | enchanter | Kayle | enchanter | Fighter | 0.593 |
| Kayle | enchanter | Nami | enchanter | Support | 0.615 |
| Lulu | enchanter | Senna | enchanter | Marksman | 0.524 |
| Milio | enchanter | Rell | engage_tank | Tank | 0.444 |
| Nami | enchanter | Kayle | enchanter | Fighter | 0.615 |
| Renata Glasc | enchanter | Alistar | engage_tank | Tank | 0.626 |
| Senna | enchanter | Lulu | enchanter | Support | 0.524 |
| Seraphine | enchanter | Zilean | enchanter | Support | 0.537 |
| Sona | enchanter | Taric | warden | Support | 0.627 |
| Soraka | enchanter | Vel'Koz | mage_support | Mage | 0.439 |
| Yuumi | enchanter | Xerath | mage_support | Mage | 0.604 |
| Zilean | enchanter | Seraphine | enchanter | Mage | 0.537 |
| Alistar | engage_tank | Renata Glasc | enchanter | Support | 0.626 |
| Amumu | engage_tank | Pyke | roamer | Support | 0.449 |
| Leona | engage_tank | Nautilus | engage_tank | Tank | 0.630 |
| Maokai | engage_tank | Tahm Kench | warden | Support | 0.375 |
| Nautilus | engage_tank | Leona | engage_tank | Tank | 0.630 |
| Pantheon | engage_tank | Neeko | mage_support | Mage | 0.608 |
| Rakan | engage_tank | Braum | warden | Support | 0.454 |
| Rell | engage_tank | Tahm Kench | warden | Support | 0.567 |
| Blitzcrank | hook | Braum | warden | Support | 0.390 |
| Thresh | hook | Morgana | catcher | Mage | 0.395 |
| Annie | mage_support | Nautilus | engage_tank | Tank | 0.396 |
| Brand | mage_support | Lulu | enchanter | Support | 0.379 |
| Fiddlesticks | mage_support | Ivern | enchanter | Support | 0.601 |
| Heimerdinger | mage_support | Fiddlesticks | mage_support | Mage | 0.486 |
| Lux | mage_support | Zyra | mage_support | Mage | 0.451 |
| Neeko | mage_support | Ivern | enchanter | Support | 0.633 |
| Swain | mage_support | Morgana | catcher | Mage | 0.570 |
| Vel'Koz | mage_support | Soraka | enchanter | Support | 0.439 |
| Xerath | mage_support | Yuumi | enchanter | Support | 0.604 |
| Zyra | mage_support | Lux | mage_support | Mage | 0.451 |
| Bard | roamer | Rell | engage_tank | Tank | 0.391 |
| Pyke | roamer | Amumu | engage_tank | Tank | 0.449 |
| Braum | warden | Nami | enchanter | Support | 0.548 |
| Tahm Kench | warden | Rell | engage_tank | Tank | 0.567 |
| Taric | warden | Sona | enchanter | Support | 0.627 |

## Figure outputs

- Projection figures were skipped because `--skip-projections` was used.
- `embedding_distance_vs_roam.png`: pairwise embedding distance vs. pairwise mean roam-score difference.
- `support_only_nearest_neighbors.png`: nearest-neighbor matrix restricted to support champions.

## Suggested wording for the report

The embedding space shows measurable continuity with empirical roaming tendency: support-pair embedding distance increases with pairwise mean roam-score difference. This supports a cautious task-specific interpretation, but it is not by itself evidence of a general semantic champion map.
