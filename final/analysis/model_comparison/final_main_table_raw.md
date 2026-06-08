# Final Main Table - Common Protocol

Metrics are recomputed from predictions on the same held-out test split. Rows are restricted to raw-target models trained under the common input protocol for learned models: 10 champion IDs + side. Practical columns `within_010` and `within_020` are the share of predictions within +/-0.10 and +/-0.20 absolute error.

## Main Table - Common Protocol Raw Models

| model                       | r2                | spearman_corr     | pearson_corr      | mae               | rmse              | pred_std          | within_010        | within_020        | n_eval | n_seeds |
| --------------------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ------ | ------- |
| Global Mean                 | -0.0008           |                   |                   | 0.1551            | 0.1906            | 0.0000            | 0.3791            | 0.6889            | 57468  | 0       |
| Champion Mean               | 0.1243            | 0.3362            | 0.3533            | 0.1438            | 0.1783            | 0.0680            | 0.4120            | 0.7306            | 57468  | 0       |
| HistGBT                     | 0.1595 +/- 0.0004 | 0.3869 +/- 0.0004 | 0.4003 +/- 0.0006 | 0.1408 +/- 0.0000 | 0.1747 +/- 0.0000 | 0.0740 +/- 0.0002 | 0.4195 +/- 0.0002 | 0.7420 +/- 0.0006 | 57468  | 3       |
| MLP OneHot                  | 0.1536 +/- 0.0010 | 0.3801 +/- 0.0005 | 0.3933 +/- 0.0006 | 0.1412 +/- 0.0000 | 0.1753 +/- 0.0001 | 0.0786 +/- 0.0007 | 0.4191 +/- 0.0007 | 0.7397 +/- 0.0008 | 57468  | 3       |
| MLP Embed Shared            | 0.1507 +/- 0.0004 | 0.3763 +/- 0.0007 | 0.3888 +/- 0.0011 | 0.1415 +/- 0.0001 | 0.1756 +/- 0.0000 | 0.0738 +/- 0.0025 | 0.4169 +/- 0.0010 | 0.7391 +/- 0.0005 | 57468  | 3       |
| MLP Per-Role + Interactions | 0.1527 +/- 0.0013 | 0.3783 +/- 0.0014 | 0.3913 +/- 0.0012 | 0.1414 +/- 0.0001 | 0.1754 +/- 0.0001 | 0.0765 +/- 0.0010 | 0.4182 +/- 0.0009 | 0.7398 +/- 0.0008 | 57468  | 3       |
