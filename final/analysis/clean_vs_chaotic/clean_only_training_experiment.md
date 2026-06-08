# Clean-Only Training Experiment

This is a secondary diagnostic, not the main model comparison table.

Both model sources are evaluated on the same held-out test split and the same feature protocol.

Validation checks:

- `clean + chaotic == all`: 42,147 + 15,321 = 57,468
- `chaos_flag` NaN count in train/test: 0 / 0
- Clean-only train rows: 197,279 of 268,322
- Feature protocol: `draft_10_champions_side`
- Seeds: 42, 123, 456

| model | subset | n | R2 | Spearman | MAE | RMSE | target_mean | pred_std |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HistGBT final weighted-train | all | 57,468 | 0.1605 | 0.3882 | 0.1408 | 0.1746 | 0.3915 | 0.0738 |
| HistGBT final weighted-train | clean | 42,147 | 0.1719 | 0.3986 | 0.1384 | 0.1711 | 0.3849 | 0.0740 |
| HistGBT final weighted-train | chaotic | 15,321 | 0.1220 | 0.3630 | 0.1473 | 0.1838 | 0.4098 | 0.0732 |
| HistGBT clean-only train | all | 57,468 | 0.1596 | 0.3876 | 0.1408 | 0.1747 | 0.3915 | 0.0751 |
| HistGBT clean-only train | clean | 42,147 | 0.1720 | 0.3983 | 0.1383 | 0.1711 | 0.3849 | 0.0753 |
| HistGBT clean-only train | chaotic | 15,321 | 0.1184 | 0.3620 | 0.1475 | 0.1842 | 0.4098 | 0.0745 |
