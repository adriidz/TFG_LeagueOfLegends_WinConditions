# ICC vs Out-of-Sample R2

- ICC: descriptive train-only consistency metric, not a test-set model score.
- R2 group mean OOS: train-only group means evaluated on test, with train global mean fallback for unseen groups.
- Compare model test R2 against `ceiling_oos_summary.csv`, not against ICC directly.
