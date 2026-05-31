# Support Roam Score v8 Productive

## Objetivo

`support_roam_score_v8_productive` redefine la etiqueta de roaming para reducir
falsos positivos producidos por caos de botlane, muertes, recalls o pathing. A
diferencia de `v5_geometry`, no permite que un caso alcance puntuaciones altas
solo por estar fuera de bot: exige evidencia productiva fuera del contexto de
botlane.

La etiqueta sigue siendo observacional y postgame. No debe usarse como feature
pregame; solo como target.

## Receta

Ventana temporal: minutos 5-12.

Componentes:

- `productive_event_score_v8`: señal fuerte. Cuenta kills/assists del support
  fuera de bot y presencia cercana del support en objetivos no-bot del equipo.
- `presence_score_v8`: señal secundaria. Ratio de frames donde el support está
  vivo, no está en base y está fuera del contexto extendido de bot.
- `xp_gap_score_v8`: señal débil heredada de v5.

Formula:

```text
productive_roam_events =
    support_kill_assists_out_bot
    + 0.5 * support_objective_presence_out_bot

productive_event_score = clip(productive_roam_events / 3, 0, 1)

raw_v8 =
    0.60 * productive_event_score
    + 0.30 * alive_outside_bot_ratio
    + 0.10 * xp_gap

score_v8 = raw_v8 ^ 0.75
```

Si `productive_roam_events == 0`, el score queda capado en `0.35`. Esto evita
que una partida rota convierta a un support pasivo en falso heavy roamer.

## Artefactos

- Script: `final/scripts/19_build_support_roam_score_v8_productive.py`
- Scores: `final/data/scores/support_scores_v8_productive_m12.parquet`
- Analisis: `final/analysis/label_v8_productive/`
- Split de entrenamiento separado: `final/data/training_v8_productive/`
- Modelo HistGBT de prueba: `final/models/gbt_v8_productive/`

## Resultados iniciales

Resumen de distribucion:

| Label | mean | std | median | q95 |
|---|---:|---:|---:|---:|
| v5 geometry | 0.392 | 0.190 | 0.388 | 0.711 |
| v8 productive | 0.358 | 0.259 | 0.321 | 0.830 |

Relación con v5:

- Pearson fila a fila v5-v8: `0.646`
- Spearman fila a fila v5-v8: `0.673`

Validación contra ranking experto:

- Spearman v8 mean vs experto: `0.810`
- Spearman v5 mean vs experto: `0.822`

Predictibilidad desde draft:

| Modelo / analisis | R2 | Spearman |
|---|---:|---:|
| Champion mean v8, val | 0.069 | 0.265 |
| HistGBT v8 raw, val | 0.091 | 0.305 |
| Ceiling botlane+side v8, train | 0.114 | - |

## Lectura metodologica

v8 es semánticamente más estricta que v5: reduce el peso de desplazamientos no
productivos y penaliza indirectamente casos como Yuumi desplazada por colapso de
botlane. Por ejemplo, Yuumi queda con media v8 `0.155`, mientras Pyke y Bard
quedan en `0.489` y `0.471`.

El coste es que v8 depende más de ejecución in-game. Por eso baja el techo
predictivo desde draft respecto a v5. Esta diferencia es útil para el informe:
v5 mide mejor predisposición espacial agregada; v8 mide mejor roaming productivo
observado, pero es menos predecible antes de empezar la partida.
