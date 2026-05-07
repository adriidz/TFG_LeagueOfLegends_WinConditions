# Explicacion del manifest previsto del tuning OAT

Este documento explica el manifest `support_oat_full_m12` antes de ejecutar los
entrenamientos en el cluster. Su objetivo es dejar claro que el tuning no es una
busqueda arbitraria, sino una continuacion directa de las conclusiones del
Informe de Progreso I: la MLP full `m12` aprende senal desde el draft, pero
presenta sobreajuste temprano y predicciones comprimidas hacia la media. Por eso
el siguiente paso es separar dos fuentes posibles de mejora: la definicion de la
etiqueta y los hiperparametros del modelo.

El manifest esta en:

```text
ProgresoActual/OAT/support_oat_full_m12/experiments/runs_manifest.csv
```

Las graficas de esta explicacion se han generado en:

```text
ProgresoActual/analysis/oat_manifest/support_oat_full_m12/
```

## Relacion con el Informe de Progreso I

El Informe de Progreso I establece tres ideas que motivan este tuning:

- El problema se reformulo de clasificacion a regresion continua porque la
  etiqueta original ya era gradual y discretizarla perdia informacion.
- La tarea se ha reducido temporalmente a support-only para validar una etiqueta
  continua defendible antes de volver a escalar a jungla/equipo.
- La primera MLP full `m12` mejora al baseline de media, pero falla en extremos:
  captura priors de campeon/composicion, aunque tiende a predecir valores
  centrales.

El tuning OAT responde a esa situacion. Si la mejora viene de cambiar pesos o
ventana de etiqueta, entonces el cuello de botella principal esta en como se
define el target. Si la mejora viene de dropout, learning rate, capacidad o
regularizacion, entonces el problema esta mas cerca del entrenamiento de la red.

## Que es el manifest

El manifest es el contrato reproducible de los experimentos. Cada fila describe
una run completa:

- que dimension cambia (`phase`, `changed_parameter`, `changed_value`);
- como se construye la etiqueta (`start_minute`, `max_minute`,
  `far_adc_threshold`, `weight_triplet`);
- que artefactos usa el entrenamiento (`support_scores_path`,
  `support_config_json`, `model_input_path`);
- donde se guardaran los resultados (`train_outdir`);
- con que hiperparametros se entrena la MLP (`batch_size`, `lr`, `hidden1`,
  `hidden2`, `dropout`, `weight_decay`, `patience`, `epochs`);
- cual sera la metrica principal de comparacion (`objective_metric=val_mse`).

La idea es que el cluster no tenga que decidir nada: cada tarea del array Slurm
lee una fila del manifest y entrena exactamente esa configuracion.

## Estructura OAT

El experimento contiene 20 runs:

| Fase | Runs | Que aisla |
|---|---:|---|
| `label_weights` | 5 | Importancia relativa de los componentes de la etiqueta |
| `time_window` | 9 | Intervalo temporal usado para medir roaming |
| `train_hparams` | 6 | Hiperparametros de entrenamiento de la MLP |

![Distribucion de runs por fase](../analysis/oat_manifest/support_oat_full_m12/manifest_phase_counts.png)

El diseno es `one-at-a-time`: cada bloque modifica una dimension manteniendo las
demas constantes. Esto evita una busqueda factorial completa y permite
interpretar los resultados. La desventaja es que no mide interacciones entre
parametros; por ejemplo, una ventana `m14` podria funcionar mejor con otro
dropout, pero esa combinacion no se prueba todavia. Para esta fase del TFG, esa
limitacion es aceptable porque el objetivo es seleccionar una configuracion
candidata y justificar decisiones, no optimizar exhaustivamente.

## Fase 1: pesos de la etiqueta

La etiqueta de support combina tres componentes:

- proporcion de frames fuera de bot extendido;
- proporcion de frames lejos del ADC;
- componente de experiencia relativa con el ADC.

La baseline del informe usa `0.45,0.35,0.20`. El manifest compara esa opcion
con variantes que enfatizan mas la salida de bot, la distancia al ADC o la
experiencia relativa.

![Tripletas de pesos](../analysis/oat_manifest/support_oat_full_m12/label_weight_triplets.png)

Esta fase responde a una pregunta metodologica: que definicion de roaming es mas
aprendible desde el draft sin dejar de ser semanticamente razonable. Si un cambio
de pesos mejora `val_mse` pero genera una distribucion menos interpretable,
habra que discutir el tradeoff entre metrica y significado de la etiqueta.

## Fase 2: ventana temporal

El Informe de Progreso I ya indica que la ventana temporal importa: support
necesitaba una ventana algo mas amplia que otras tareas en los experimentos
anteriores. Por eso el OAT prueba combinaciones de minuto inicial `4, 5, 6` y
minuto final `10, 12, 14`, manteniendo los pesos baseline.

![Grid de ventanas](../analysis/oat_manifest/support_oat_full_m12/time_window_grid.png)

La ventana baseline es `s5_m12`. Las ventanas mas cortas pueden reducir ruido de
mid game, pero pueden perder roams tardios hacia objetivos. Las ventanas mas
largas capturan mas comportamiento, pero tambien pueden mezclar decisiones
derivadas del estado de la partida, no solo de la predisposicion del draft.

## Fase 3: hiperparametros de la MLP

La fase de hiperparametros mantiene la etiqueta baseline y cambia una variable
de entrenamiento cada vez. Esto permite aislar si la compresion de predicciones
observada en la MLP full mejora con regularizacion, capacidad o dinamica de
aprendizaje.

![Variantes de hiperparametros](../analysis/oat_manifest/support_oat_full_m12/train_hparams_variants.png)

La baseline de red es:

| Elemento | Valor |
|---|---:|
| Feature groups | `standard` |
| Batch size | 256 |
| Epochs | 60 |
| Learning rate | 1e-3 |
| Capas ocultas | 256 -> 128 |
| Dropout | 0.2 |
| Weight decay | 1e-5 |
| Patience | 10 |
| Split validacion | 0.2 |
| Seed | 42 |

Las variantes previstas son: `lr=5e-4`, `dropout=0.1`, `dropout=0.3`,
`hidden=512-256`, `weight_decay=1e-4` y `batch_size=512`.

## Salud de etiqueta antes de entrenar

Aunque el entrenamiento no se haya ejecutado todavia, el manifest ya contiene
artefactos de etiqueta y model input. Por tanto se puede revisar si las runs
previstas tienen una escala y cobertura razonables antes de gastar GPU.

![Estadisticos de etiqueta por run](../analysis/oat_manifest/support_oat_full_m12/planned_label_stats_by_run.png)

Resumen de las 20 runs:

| Metrica | Valor |
|---|---:|
| Runs totales | 20 |
| Epocas previstas por run | 60 |
| Minimo de filas etiquetadas | 334852 |
| Maximo de filas etiquetadas | 337120 |
| Media minima del target | 0.2497 |
| Media maxima del target | 0.3219 |
| Metrica objetivo | `val_mse` |

La variacion de medias es esperable: cambiar pesos o ventana desplaza la escala
del target. Lo importante es que ninguna configuracion parece degenerada antes
de entrenar: todas mantienen cientos de miles de observaciones y una variacion
continua del score.

## Huella de artefactos

Como el cluster recibira estos inputs por Git, no por sincronizacion manual, el
manifest y sus artefactos se guardan bajo `ProgresoActual/OAT`. Los modelos
entrenados no entran aqui: se escribiran en `ProgresoActual/models/oat_tuning/`
y se sincronizaran de vuelta desde el cluster.

![Huella de artefactos por fase](../analysis/oat_manifest/support_oat_full_m12/artifact_footprint_by_phase.png)

Los artefactos principales ocupan aproximadamente:

| Tipo | Tamano |
|---|---:|
| `support_scores.parquet` | 281.3 MB |
| `model_input.parquet` | 626.4 MB |

Esta huella es el coste de hacer que el cluster pueda entrenar solo con
`git pull`, sin raw local ni reconstruccion de etiquetas en remoto.

## Como se interpretaran los resultados

Cuando el cluster vuelva a estar disponible, cada run generara metricas en su
`train_outdir`. La comparacion principal sera por `val_mse`, pero no deberia
leerse de forma aislada. Para seleccionar una configuracion candidata conviene
mirar:

- `val_mse` y `mae`, como error de regresion;
- `r2`, `pearson_corr` y `spearman_corr`, como senal y orden relativo;
- curvas de train/validacion, para detectar sobreajuste temprano;
- scatter true-vs-pred y error por bins, para comprobar si se reduce la
  compresion hacia la media;
- distribucion de la etiqueta, para no elegir una mejora numerica que rompa la
  interpretabilidad del target.

La configuracion final no deberia defenderse solo como la que minimiza `val_mse`,
sino como la que equilibra tres criterios: target semanticamente razonable,
mejor aprendizaje desde draft y diagnosticos visuales coherentes.

## Artefactos generados

| Artefacto | Ruta |
|---|---|
| Resumen CSV del manifest | `ProgresoActual/analysis/oat_manifest/support_oat_full_m12/oat_manifest_planned_summary.csv` |
| Resumen JSON | `ProgresoActual/analysis/oat_manifest/support_oat_full_m12/oat_manifest_planned_summary.json` |
| Runs por fase | `ProgresoActual/analysis/oat_manifest/support_oat_full_m12/manifest_phase_counts.png` |
| Pesos de etiqueta | `ProgresoActual/analysis/oat_manifest/support_oat_full_m12/label_weight_triplets.png` |
| Ventanas temporales | `ProgresoActual/analysis/oat_manifest/support_oat_full_m12/time_window_grid.png` |
| Hiperparametros | `ProgresoActual/analysis/oat_manifest/support_oat_full_m12/train_hparams_variants.png` |
| Estadisticos de etiqueta | `ProgresoActual/analysis/oat_manifest/support_oat_full_m12/planned_label_stats_by_run.png` |
| Huella de artefactos | `ProgresoActual/analysis/oat_manifest/support_oat_full_m12/artifact_footprint_by_phase.png` |
