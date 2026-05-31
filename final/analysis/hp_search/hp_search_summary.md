# MLP HP Search Summary

- Configurations evaluated: 108
- Default validation Spearman: 0.372226
- Best run: `020_h128-64_d0p4_lr1em03_wd5em04`
- Best validation Spearman: 0.376688
- Best validation R2: 0.150537
- Hidden dims: [128, 64]
- Dropout: 0.4
- LR: 0.001
- Weight decay: 0.0005
- Delta vs default: 0.004462

Decision: la MLP es robusta a la configuracion; se conserva el default porque la mejora no supera el umbral de 0.005 Spearman.

## Test Evaluation Of Selected Run
- MLP Per-Role + Interactions HP Best [raw]: R2=0.154677, Spearman=0.383654
