# Clean vs Chaotic - HistGBT Final Model (Test Set)

Model source: HistGBT final retrained seed artifacts, averaged at prediction time.

Chaos flag definition:

- `support_deaths_0_12 + adc_deaths_0_12 >= 6`
- `adc_deaths_0_12 >= 5`
- `support_deaths_0_12 >= 4 AND support_kill_assists_out_bot_0_12 == 0`

Validation checks:

- `clean + chaotic == all`: 42,147 + 15,321 = 57,468
- `chaos_flag` NaN count: 0
- Feature protocol: `draft_10_champions_side`
- Model seeds used: 123, 42, 456

| subset | n | R2 | Spearman | MAE | RMSE | target_mean | pred_std |
|--------|---|----| ---------|-----|------|-------------|----------|
| all | 57,468 | 0.1605 | 0.3882 | 0.1408 | 0.1746 | 0.3915 | 0.0738 |
| clean | 42,147 | 0.1719 | 0.3986 | 0.1384 | 0.1711 | 0.3849 | 0.0740 |
| chaotic | 15,321 | 0.1220 | 0.3630 | 0.1473 | 0.1838 | 0.4098 | 0.0732 |
