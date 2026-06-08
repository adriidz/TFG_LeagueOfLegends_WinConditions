# Embedding analysis summary

## Dataset

- Embedding label: `missing_meta_ally_utility`
- Embedding file analyzed: `C:\Users\adria\Desktop\TFG\final\models\mlp_per_role\embeddings_raw_ally_utility.json`
- Champions with embeddings: 172
- Support-labeled champions: 0
- Embedding dimension: 16

Support archetype counts:

- No support archetype metadata available.

Primary Data Dragon class counts:

- Unknown: 172

## Cluster coherence

- Support archetype silhouette, cosine metric: n/a (n=0, groups=0)
- Primary class silhouette, cosine metric: n/a (n=172, groups=1)

Interpretation guide: silhouette near 1 implies compact, separated clusters; near 0 implies overlapping groups; negative values imply that many samples are closer to other groups than to their own.

## Roam-score continuity

- Roam-score continuity analysis skipped; no training means were available.

## Nearest-neighbor structure

### Neighbors among all champions (Exploratory / unrestricted)
These neighbors are useful for inspection, but they can mix support and non-support champions and should not be used as the main support-archetype check.
- Top-5 neighbors that are support-labeled: n/a
- Top-5 neighbors with same support archetype: n/a
- Top-5 neighbors with same primary class: n/a
- Rank-1 nearest neighbor support-labeled rate: n/a
- Rank-1 same support archetype rate: n/a

### Neighbors restricted to support champions (Main support-only check)
- Top-5 support neighbors with same support archetype: n/a
- Top-5 support neighbors with same primary class: n/a
- Rank-1 nearest support neighbor with same support archetype rate: n/a

Rank-1 nearest neighbors for support champions (all champions, exploratory):

| Support | Archetype | Nearest neighbor | Neighbor archetype | Neighbor class | Cosine sim. |
|---|---|---|---|---|---:|
| n/a | n/a | n/a | n/a | n/a | n/a |

Rank-1 nearest neighbors restricted to support champions:

| Support | Archetype | Nearest support neighbor | Neighbor archetype | Neighbor class | Cosine sim. |
|---|---|---|---|---|---:|
| n/a | n/a | n/a | n/a | n/a | n/a |

## Figure outputs

- Projection figures were skipped because `--skip-projections` was used.

## Suggested wording for the report

These checks do not support a strong semantic-embedding claim. The embeddings may still encode task-specific signal for prediction, but the observed support-only neighborhood and archetype-alignment metrics are not enough to describe the space as a clean semantic map.
