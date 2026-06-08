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
| MLP OneHot                                  | raw            | 0.3807        | 0.4187     | 0.7405     | 0.4836        | 0.2451             | 0.2292                | 0.1766        | 0.3280                      | 0.2922                         | 0.2857                 |
| MLP OneHot                                  | raw            | 0.3795        | 0.4200     | 0.7386     | 0.4844        | 0.2429             | 0.2270                | 0.1733        | 0.3248                      | 0.2898                         | 0.2816                 |
| MLP Per-Role + Interactions                 | raw            | 0.3780        | 0.4192     | 0.7400     | 0.4832        | 0.2399             | 0.2245                | 0.1696        | 0.3276                      | 0.2926                         | 0.2818                 |
| MLP Per-Role + Interactions                 | raw            | 0.3802        | 0.4184     | 0.7407     | 0.4831        | 0.2350             | 0.2200                | 0.1633        | 0.3259                      | 0.2914                         | 0.2789                 |
| MLP Embed Shared                            | raw            | 0.3758        | 0.4174     | 0.7395     | 0.4824        | 0.2328             | 0.2179                | 0.1605        | 0.3177                      | 0.2843                         | 0.2707                 |
| MLP OneHot                                  | raw            | 0.3799        | 0.4185     | 0.7399     | 0.4831        | 0.2329             | 0.2175                | 0.1605        | 0.3251                      | 0.2903                         | 0.2791                 |
| MLP Per-Role + Interactions                 | raw            | 0.3768        | 0.4169     | 0.7388     | 0.4828        | 0.2316             | 0.2169                | 0.1563        | 0.3190                      | 0.2858                         | 0.2709                 |
| HistGBT                                     | raw            | 0.3870        | 0.4196     | 0.7416     | 0.4826        | 0.2303             | 0.2156                | 0.1523        | 0.3272                      | 0.2936                         | 0.2759                 |
| HistGBT                                     | raw            | 0.3864        | 0.4192     | 0.7417     | 0.4829        | 0.2296             | 0.2149                | 0.1512        | 0.3276                      | 0.2940                         | 0.2762                 |
| HistGBT                                     | raw            | 0.3874        | 0.4196     | 0.7428     | 0.4829        | 0.2300             | 0.2153                | 0.1512        | 0.3263                      | 0.2930                         | 0.2740                 |
| MLP Embed Shared                            | raw            | 0.3759        | 0.4179     | 0.7384     | 0.4827        | 0.2249             | 0.2105                | 0.1491        | 0.3164                      | 0.2839                         | 0.2662                 |
| Champion Mean                               | raw            | 0.3362        | 0.4120     | 0.7306     | 0.4795        | 0.2101             | 0.1968                | 0.1413        | 0.2878                      | 0.2600                         | 0.2340                 |
| MLP Embed Shared                            | raw            | 0.3773        | 0.4156     | 0.7393     | 0.4810        | 0.2115             | 0.1980                | 0.1295        | 0.3153                      | 0.2843                         | 0.2591                 |
| MLP OneHot (quantile->raw)                  | quantile       | 0.3781        | 0.4146     | 0.7362     | 0.4798        | 0.1964             | 0.1840                | 0.1046        | 0.3165                      | 0.2863                         | 0.2561                 |
| MLP Per-Role + Interactions (quantile->raw) | quantile       | 0.3786        | 0.4137     | 0.7356     | 0.4795        | 0.1896             | 0.1777                | 0.0985        | 0.3131                      | 0.2838                         | 0.2503                 |
| MLP Per-Role + Interactions (quantile->raw) | quantile       | 0.3804        | 0.4131     | 0.7353     | 0.4782        | 0.1843             | 0.1728                | 0.0935        | 0.3167                      | 0.2871                         | 0.2532                 |
| MLP Per-Role + Interactions (quantile->raw) | quantile       | 0.3794        | 0.4152     | 0.7368     | 0.4782        | 0.1809             | 0.1695                | 0.0894        | 0.3137                      | 0.2845                         | 0.2495                 |
| MLP OneHot (quantile->raw)                  | quantile       | 0.3800        | 0.4128     | 0.7357     | 0.4770        | 0.1775             | 0.1664                | 0.0856        | 0.3119                      | 0.2828                         | 0.2477                 |
| HistGBT (quantile->raw)                     | quantile       | 0.3873        | 0.4155     | 0.7375     | 0.4778        | 0.1799             | 0.1687                | 0.0840        | 0.3197                      | 0.2904                         | 0.2518                 |
| HistGBT (quantile->raw)                     | quantile       | 0.3873        | 0.4144     | 0.7375     | 0.4780        | 0.1796             | 0.1683                | 0.0834        | 0.3176                      | 0.2884                         | 0.2505                 |
| HistGBT (quantile->raw)                     | quantile       | 0.3870        | 0.4149     | 0.7377     | 0.4780        | 0.1784             | 0.1673                | 0.0822        | 0.3187                      | 0.2895                         | 0.2506                 |
| MLP Embed Shared (quantile->raw)            | quantile       | 0.3773        | 0.4119     | 0.7349     | 0.4759        | 0.1601             | 0.1501                | 0.0672        | 0.3104                      | 0.2821                         | 0.2431                 |
| MLP OneHot (quantile->raw)                  | quantile       | 0.3787        | 0.4131     | 0.7341     | 0.4758        | 0.1593             | 0.1494                | 0.0645        | 0.3147                      | 0.2858                         | 0.2447                 |
| MLP Embed Shared (quantile->raw)            | quantile       | 0.3783        | 0.4115     | 0.7340     | 0.4763        | 0.1588             | 0.1489                | 0.0632        | 0.3112                      | 0.2830                         | 0.2435                 |
| MLP Embed Shared (quantile->raw)            | quantile       | 0.3760        | 0.4114     | 0.7337     | 0.4753        | 0.1528             | 0.1433                | 0.0556        | 0.3099                      | 0.2820                         | 0.2414                 |
| Champion Mean (quantile->raw)               | quantile       | 0.3361        | 0.4081     | 0.7274     | 0.4732        | 0.1267             | 0.1189                | 0.0351        | 0.2831                      | 0.2576                         | 0.2207                 |
| Global Mean                                 | raw            | -             | 0.3791     | 0.6889     | 0.4651        | -                  | -                     | -             | -                           | -                              | -                      |
| Global Mean (quantile->raw)                 | quantile       | -             | 0.3774     | 0.6869     | 0.4651        | -                  | -                     | -             | -                           | -                              | -                      |


## Reading

The top row by fixed-bin QWK is **MLP OneHot** (raw). Its fixed-bin QWK is 0.1766, compared with continuous Spearman 0.3807. This should be read as strategic zone agreement, not exact score recovery.

If QWK or bin correlations are meaningfully higher than exact-score metrics, the model is more useful as an ordinal coach signal than as a precise regressor. If Champion Mean remains close to GBT, most of the ordinal signal is already carried by support champion identity.
