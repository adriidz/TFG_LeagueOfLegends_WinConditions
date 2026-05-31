# Calidad de la etiqueta y filtrado de ruido

Documento de decisión sobre la calidad de `support_roam_score` y el
filtrado de partidas caóticas. Fecha: 15 mayo 2026.

## 1. Qué mide la etiqueta actual (v5)

La etiqueta `support_roam_score` es una variable continua en [0, 1] que
resume la separación observada entre el support y el contexto de botlane
durante el early-game (minutos 5-12).

Receta:

```
score_raw = 0.45 × outside_ratio + 0.35 × far_ratio + 0.20 × xp_gap
score = score_raw ^ 0.75
```

- `outside_ratio`: fracción de frames donde el support está fuera de bot
  context (BOT_LANE_CORE, BOT_SIDE_NEAR, RIVER_BOT, DRAGON_AREA, bases).
- `far_ratio`: fracción de frames cooperativos (ambos vivos, ADC fuera de
  base) donde la distancia support-ADC es ≥ 2500 unidades.
- `xp_gap`: normalización inversa del ratio XP support/ADC al minuto 12.

Fuente de datos: timeline frames de Riot (snapshots minutales). Del minuto
5 al 12 hay como máximo 8 frames. Los frames con support muerto o en base
se descartan.

## 2. Problemas identificados

### 2.1 La etiqueta mide separación, no intención

La etiqueta no distingue entre:
- Un support que rota a mid para una gank (roaming real).
- Un support que muere repetidamente y cuyos frames muestran posiciones
  dispersas (caos de botlane).
- Un support cuyo ADC muere y queda solo en lane.

Los tres escenarios producen outside_ratio y far_ratio altos.

### 2.2 Resolución temporal baja

Con 8 snapshots máximo (5-7 típicos tras filtrar muertes y bases), cada
frame vale entre 14% y 20% del score. Un solo frame atípico puede cambiar
el outside_ratio de 0.60 a 0.80.

Esta limitación es inherente a la API de Riot, que solo proporciona
posiciones a nivel de minuto. No se puede resolver con más datos.

### 2.3 Partidas caóticas contaminan la etiqueta

La auditoría cualitativa (`analysis/qualitative_case_audit/`) reveló que
17 de 20 top errors del modelo tienen el tag `chaotic_early_game`:

- Support deaths 0-12: 2-6 muertes.
- ADC deaths 0-12: 1-8 muertes.
- Score = 0.8-1.0.

En estas partidas, la botlane colapsó. No hubo una decisión de roamear;
hubo un snowball violento que separó a los jugadores. El draft no puede
predecir este tipo de ejecución.

### 2.4 XP gap en partidas caóticas

Cuando el support muere varias veces, pierde XP. El xp_gap sube, pero no
por roaming sino por feeding. Esto contribuye al score erróneamente.

## 3. Evaluación: ¿es la etiqueta mala?

No. La etiqueta captura señal real de roaming:
- Correlación Spearman 0.82 con referencia experta por campeón.
- Los campeones con score alto son roamers conocidos (Bard, Pyke, Alistar).
- Los campeones con score bajo son enchanters pegados a ADC (Yuumi, Lulu).

El problema no es la definición del score sino la contaminación por
outliers caóticos.

## 4. Intentos anteriores de mejora y resultado

Se construyeron dos variantes experimentales:

- **v6 (scripts 13-14)**: añade evidencia de eventos (combate fuera de bot,
  visión) como canales adicionales con pesos configurables. 15 variantes
  probadas con sweep. Mejor mejora: +0.002 Spearman.
- **v7 (script 15)**: usa posiciones de kill/muerte como muestras espaciales
  extras para aumentar resolución. Correlación v5↔v7 ≈ 0.99.

Conclusión: cambiar la fórmula no cambia la señal. Con 8 frames minutales,
cualquier combinación lineal de outside/far/xp mide esencialmente lo mismo.

## 5. Decisión: filtrado de ruido por caos

En lugar de redefinir la etiqueta, se filtra el ruido de entrenamiento.

### 5.1 Chaos flag

Se define un indicador binario `chaos_flag` a partir de eventos tempranos
(datos de `12_build_support_event_context.py`):

```python
chaos_flag = (
    (support_deaths_0_12 + adc_deaths_0_12 >= 6)
    | (adc_deaths_0_12 >= 5)
    | (
        (support_deaths_0_12 >= 4)
        & (support_kill_assists_out_bot_0_12 == 0)
    )
)
```

La lógica es:
- Botlane combinada muere 6+ veces → caos evidente.
- ADC muere 5+ veces → botlane colapsó.
- Support muere 4+ veces sin ninguna acción activa fuera de bot → feeding,
  no roaming.

### 5.2 Sample weight

Las observaciones marcadas como caóticas reciben `sample_weight = 0.2`
durante entrenamiento. Las observaciones limpias reciben `sample_weight = 1.0`.

Esto reduce la influencia de partidas donde el score alto no refleja
predisposición de draft sino ejecución caótica.

### 5.3 Min support frames

Se sube `min_support_frames` de 2 a 3. Partidas con solo 2 frames válidos
producen outside_ratio binario (0/0.5/1.0), que es demasiado ruidoso.

El filtro se aplica al generar los splits de entrenamiento. Las partidas con
< 3 frames se excluyen completamente.

### 5.4 Confidence actualizada

```python
confidence_final = min(1, valid_support_frames / 6) × (1 - 0.3 × chaos_flag)
```

## 6. Implementación

Script: `final/scripts/16_add_chaos_filter_weights.py`.

Entrada:
- `final/data/training/{train,val,test}.parquet` (splits existentes).
- `final/data/scores/support_scores_v5_geometry_m12.parquet` (para
  valid_support_frames_v5).
- `final/data/event_context/support_event_context_m12.parquet` (para
  muertes tempranas y evidencia de actividad fuera de bot).

Salida:
- Splits actualizados con columnas adicionales: `chaos_flag`,
  `sample_weight`, `valid_support_frames_v5`, `confidence_final`.
- Partidas con < 3 frames válidos excluidas.
- `final/data/training/chaos_filter_summary.json` con estadísticas.

## 7. Uso en modelos

- `HistGradientBoostingRegressor.fit(X, y, sample_weight=weights)`.
- MLP: loss ponderado por sample_weight en el training loop.
- Comparación: modelo con weights vs modelo sin weights es un resultado
  presentable.

## 8. Narrativa para la memoria

> La etiqueta `support_roam_score` captura separación observada entre
> support y ADC durante el early-game. La resolución de ~8 snapshots por
> partida es una limitación inherente de la fuente de datos. Para mitigar
> la contaminación por partidas caóticas, se aplica un filtro basado en
> muertes tempranas de la botlane que reduce el peso de observaciones
> ruidosas durante el entrenamiento.
