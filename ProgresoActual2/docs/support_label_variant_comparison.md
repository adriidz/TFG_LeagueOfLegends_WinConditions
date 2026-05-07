# Comparacion de variantes de etiqueta support

Esta carpeta contiene la segunda linea de progreso del TFG. El objetivo es
comparar variantes CPU-only de la etiqueta `support_roam_score` sin entrenar la
MLP.

La entrada principal se lee desde:

```text
ProgresoActual/data/clean/frame_state/support_frame_state.parquet
```

La referencia experta se usa solo como validacion externa de ranking por
campeon. No se usa para construir la etiqueta.

## Smoke reproducible

El smoke no reutiliza el `sample5` historico. Toma un 5% aleatorio de `match_id`
desde el frame-state full, con `seed=42`, conservando ambos equipos y todos los
frames de cada partida seleccionada.

```powershell
.\ProgresoActual2\scripts\run_support_label_variant_comparison.ps1 -Mode smoke
```

## Full y export

Cuando el smoke este revisado:

```powershell
.\ProgresoActual2\scripts\run_support_label_variant_comparison.ps1 -Mode full -ExportSelected
```

Esto genera:

```text
ProgresoActual2/analysis/support_label_variants/full/
ProgresoActual2/data/clean/scores/support_scores_v3_m12.parquet
ProgresoActual2/data/clean/scores/selected_support_score_v3_config.json
```

## Criterio

La candidata debe mantener cobertura alta, correlacion Spearman >= 0.80 contra
la referencia experta, ampliar el rango util frente a v2 y no saturar
artificialmente el extremo 1. Si ninguna variante cumple todas las reglas, se
exporta la mejor alternativa por ranking compuesto y queda marcado en el JSON.
