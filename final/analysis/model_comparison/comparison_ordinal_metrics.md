# Ordinal Metrics for Strategic Utility

These metrics complement exact regression metrics. They evaluate whether the model places a draft in a sensible strategic roaming zone rather than requiring an exact decimal score.

- **Continuous Spearman** (`spearman_corr`) ranks exact raw scores and predictions.
- **Bin Spearman** ranks ordinal bins, ignoring small differences inside the same bin.
- **Bin Kendall tau** measures pairwise ordinal agreement between true and predicted bins.
- **Quadratic Weighted Kappa (QWK)** measures ordinal agreement while penalizing distant bin errors more than adjacent bin errors.

All ordinal metrics are computed on raw-scale bins. Quantile-trained models are inverse-transformed to raw before evaluation.

## Ordinal Metric Comparison

| model                                       | trained_target | spearman_corr | within_010 | within_020 | fixed_bin_acc | fixed_bin_spearman | fixed_bin_kendall_tau | fixed_bin_qwk | train_quantile_bin_spearman | train_quantile_bin_kendall_tau | train_quantile_bin_qwk |
| ------------------------------------------- | -------------- | ------------- | ---------- | ---------- | ------------- | ------------------ | --------------------- | ------------- | --------------------------- | ------------------------------ | ---------------------- |
| MLP OneHot                                  | raw            | 0.3807        | 0.4185     | 0.7410     | 0.4836        | 0.2460             | 0.2300                | 0.1779        | 0.3269                      | 0.2911                         | 0.2855                 |
| MLP Per-Role + Interactions                 | raw            | 0.3806        | 0.4176     | 0.7413     | 0.4829        | 0.2366             | 0.2215                | 0.1657        | 0.3251                      | 0.2907                         | 0.2788                 |
| HistGBT + Pair TE                           | raw            | 0.3882        | 0.4183     | 0.7419     | 0.4836        | 0.2371             | 0.2220                | 0.1609        | 0.3285                      | 0.2947                         | 0.2779                 |
| HistGBT + Archetypes                        | raw            | 0.3881        | 0.4181     | 0.7405     | 0.4826        | 0.2338             | 0.2190                | 0.1578        | 0.3281                      | 0.2944                         | 0.2776                 |
| HistGBT                                     | raw            | 0.3874        | 0.4185     | 0.7415     | 0.4831        | 0.2314             | 0.2166                | 0.1538        | 0.3290                      | 0.2952                         | 0.2780                 |
| Champion Mean                               | raw            | 0.3360        | 0.4109     | 0.7291     | 0.4794        | 0.2098             | 0.1965                | 0.1413        | 0.2879                      | 0.2602                         | 0.2340                 |
| MLP Embed                                   | raw            | 0.3755        | 0.4184     | 0.7381     | 0.4818        | 0.2174             | 0.2035                | 0.1389        | 0.3155                      | 0.2834                         | 0.2639                 |
| MLP Per-Role + Interactions (quantile->raw) | quantile       | 0.3792        | 0.4145     | 0.7362     | 0.4797        | 0.1929             | 0.1808                | 0.1021        | 0.3148                      | 0.2852                         | 0.2522                 |
| HistGBT + Archetypes (quantile->raw)        | quantile       | 0.3873        | 0.4147     | 0.7359     | 0.4783        | 0.1842             | 0.1727                | 0.0889        | 0.3176                      | 0.2882                         | 0.2518                 |
| HistGBT + Pair TE (quantile->raw)           | quantile       | 0.3884        | 0.4146     | 0.7363     | 0.4783        | 0.1842             | 0.1728                | 0.0873        | 0.3167                      | 0.2877                         | 0.2502                 |
| MLP Embed (quantile->raw)                   | quantile       | 0.3772        | 0.4154     | 0.7364     | 0.4782        | 0.1776             | 0.1665                | 0.0866        | 0.3122                      | 0.2833                         | 0.2483                 |
| HistGBT (quantile->raw)                     | quantile       | 0.3871        | 0.4155     | 0.7375     | 0.4780        | 0.1820             | 0.1706                | 0.0862        | 0.3181                      | 0.2888                         | 0.2509                 |
| MLP OneHot (quantile->raw)                  | quantile       | 0.3786        | 0.4131     | 0.7341     | 0.4765        | 0.1621             | 0.1520                | 0.0668        | 0.3140                      | 0.2852                         | 0.2445                 |
| Champion Mean (quantile->raw)               | quantile       | 0.3362        | 0.4064     | 0.7260     | 0.4732        | 0.1268             | 0.1190                | 0.0346        | 0.2832                      | 0.2577                         | 0.2207                 |
| Global Mean (quantile->raw)                 | quantile       | -             | 0.3790     | 0.6883     | 0.4651        | -                  | -                     | -             | -                           | -                              | -                      |
| Global Mean                                 | raw            | -             | 0.3781     | 0.6882     | 0.4651        | -                  | -                     | -             | -                           | -                              | -                      |


## Reading

The top row by fixed-bin QWK is **MLP OneHot** (raw). Its fixed-bin QWK is 0.1779, compared with continuous Spearman 0.3807. This should be read as strategic zone agreement, not exact score recovery.

If QWK or bin correlations are meaningfully higher than exact-score metrics, the model is more useful as an ordinal coach signal than as a precise regressor. If Champion Mean remains close to GBT, most of the ordinal signal is already carried by support champion identity.
