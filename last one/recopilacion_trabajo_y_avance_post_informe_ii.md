# Recopilacion del trabajo realizado y avance post-Informe II

Fecha de recopilacion: 2026-06-09

Fuentes revisadas:

- `PropuestaInicial/docs/TFG Inferencia de Early Game en League of Legends.pdf`
- `ProgresoActual/docs/Informe de Progreso I.pdf`
- `final/docs/Informe de Progreso II.pdf`
- `last one/plan_accion_tfg.md`
- Historial de git desde `20df30589` (`entrevista lunes 1/06`, commit donde entra el Informe de Progreso II)
- Cambios locales no confirmados del repositorio a 2026-06-09
- Artefactos recientes en `final/analysis`, `final/scripts`, `tests`, `last one` y `final/docs`

## 1. Evolucion general del TFG

El TFG empezo como un sistema para inferir tendencias tempranas en League of Legends usando exclusivamente informacion disponible antes de empezar la partida. La propuesta inicial planteaba tres salidas: comportamiento del jungler, roaming del support y tendencia espacial del equipo. La formulacion original era una clasificacion multi-output con etiquetas discretizadas a partir de indices continuos calculados retrospectivamente desde la timeline.

Durante el desarrollo, el proyecto se reoriento de forma importante. El primer bloque experimental mostro que la clasificacion introducia problemas artificiales: la clase intermedia o ambigua concentraba muchos casos dificiles, eliminarla mejoraba metricas pero descartaba informacion, y los errores entre clases cercanas se penalizaban igual que errores entre extremos. Por eso, el Informe de Progreso I reformula el trabajo como regresion continua y centra el alcance en una primera tarea: predecir `support_roam_score`, un score continuo de movilidad temprana del agente de apoyo.

El Informe de Progreso II consolida ese cambio: el objetivo deja de ser "predecir exactamente el roaming" y pasa a ser cuantificar cuanta senal contiene la configuracion prepartida sobre la movilidad temprana del support. Esta lectura es mas fuerte academicamente: el resultado principal no es solo un modelo, sino una estimacion del limite predictivo de la informacion prepartida.

## 2. Trabajo acumulado hasta el Informe de Progreso II

### 2.1 Datos e infraestructura

Se construyo un pipeline propio a partir de la API de Riot Games y Data Dragon. La unidad de analisis final es `(match_id, team_id)`, de modo que cada partida puede aportar dos observaciones, una por equipo. El proyecto separa estrictamente:

- Variables de entrada: draft y metadatos disponibles antes del minuto cero.
- Etiqueta: comportamiento observado en la timeline durante los primeros minutos.

En el Informe I ya existian:

- recoleccion de partidas y timelines;
- transformacion de JSONs en tablas de draft y frame-state;
- calculo inicial de scores de jungler, support y equipo;
- experimentos de discretizacion y clasificacion;
- analisis de ventanas temporales;
- primera MLP de regresion para support.

En el Informe II el dataset final reportado pasa a 383.247 observaciones partida-equipo, procedentes de unas 191.000 partidas, EUW, Ranked Solo/Duo, Master/Grandmaster/Challenger, parches 16.2 a 16.8.

### 2.2 Etiqueta `support_roam_score`

La etiqueta mide movilidad temprana del support entre los minutos 5 y 12. No existe ground truth oficial de "roaming", asi que se construyo una proxy a partir de la timeline.

Formula principal del Informe II:

```text
score_raw = 0.45 * outside_ratio + 0.35 * far_ratio + 0.20 * xp_gap
score = score_raw ^ 0.75
```

Componentes:

- `outside_ratio`: proporcion de frames en los que el support esta fuera del contexto de botlane.
- `far_ratio`: proporcion de frames en los que el support esta lejos del ADC/botlaner.
- `xp_gap`: diferencia relativa de experiencia entre support y botlaner.

Tambien se reviso la geometria del mapa. La geometria automatica inicial fue reemplazada por una geometria manual mas interpretable, con zonas como botlane, rio inferior, dragon, midlane, junglas por cuadrante y bases. Esta decision mejora la defendibilidad de la etiqueta porque el score depende directamente de que posiciones se consideren fuera de la zona asignada.

La etiqueta se valido de forma cualitativa con una referencia experta manual de 47 campeones support. El resultado fue fuerte: Pearson ~= 0.795 y Spearman ~= 0.825 entre el ranking experto y el score medio observado por campeon. Esto no convierte la etiqueta en ground truth, pero si demuestra que no es ruido arbitrario.

### 2.3 Modelos entrenados y comparados hasta el Informe II

El Informe II compara:

- baselines: media global y media historica por campeon support;
- HistGradientBoosting base;
- HistGBT con arquetipos;
- HistGBT con Pair Target Encoding;
- MLP OneHot;
- MLP con embeddings compartidos;
- MLP con embeddings por rol e interacciones;
- variantes quantile de la etiqueta;
- variantes de etiqueta orientadas a eventos;
- grid de hiperparametros de 108 configuraciones para MLP.

Resultados principales reportados en el Informe II:

| Modelo / referencia | R2 test | Spearman | MAE |
|---|---:|---:|---:|
| ICC / referencia botlane+side del informe | 0.173 | - | - |
| HistGBT + Pair Target | 0.161 | 0.388 | 0.141 |
| HistGBT + Archetypes | 0.161 | 0.388 | 0.141 |
| HistGBT base | 0.160 | 0.387 | 0.141 |
| MLP OneHot | 0.155 | 0.381 | 0.141 |
| MLP Per-Role + Interactions | 0.154 | 0.381 | 0.141 |
| MLP Embeddings | 0.150 | 0.376 | 0.142 |
| Champion Mean | 0.125 | 0.336 | 0.144 |
| Global Mean | 0.000 | - | 0.155 |

Conclusiones hasta el Informe II:

- El draft contiene senal real, pero parcial.
- La mayor parte de la senal predecible esta en el propio campeon support.
- El resto del draft aporta informacion, pero con margen limitado.
- Los modelos tabulares funcionan ligeramente mejor que las MLPs.
- Las MLPs no mejoran con embeddings ni con un grid amplio de hiperparametros.
- Las predicciones estan comprimidas hacia la media: el modelo ordena mejor de lo que calibra extremos.
- Las partidas caoticas explican parte importante del error.

### 2.4 Partidas caoticas y auditoria de errores

El Informe II introduce `chaos_flag` y `sample_weight`:

- Partidas limpias: `sample_weight = 1.0`
- Partidas caoticas: `sample_weight = 0.2`

Una observacion se marca como caotica si hay senales tempranas anormales de colapso de botlane, como muchas muertes del botlaner/support antes del minuto 12 o pocos frames validos. Aproximadamente el 26.5% de observaciones quedan marcadas como caoticas.

Resultado clave del Informe II:

- HistGBT en partidas limpias: R2 ~= 0.171
- HistGBT en partidas caoticas: R2 ~= 0.122

Interpretacion: el draft permite estimar predisposicion, pero no puede anticipar completamente ejecucion anomala, decisiones individuales, errores, frustracion o comportamientos no cooperativos.

### 2.5 Prototipo CLI

Hasta el Informe II ya existia una version funcional del prototipo por terminal:

- acepta composiciones de draft;
- completa hechizos de invocador por defecto si faltan;
- carga el modelo final;
- devuelve score e interpretacion;
- puede ejecutarse en modo interactivo, por argumentos o por batch.

Quedaba pendiente pulir salida interpretativa y dejarlo preparado para entrega final.

## 3. Plan de accion posterior al Informe II

El plan `last one/plan_accion_tfg.md` identifica cuatro criticas principales del tutor:

1. Falta de rigor experimental y comparaciones no plenamente justas.
2. Confusion entre ICC y R2.
3. Exceso de vocabulario especifico de League of Legends.
4. Explicacion insuficiente de embeddings.

El plan propone:

- reentrenar/comparar modelos con un protocolo comun de features: 10 campeones + side;
- asegurar `sample_weight` en todos los modelos principales;
- usar 3 seeds y reportar media +/- desviacion;
- integrar WandB;
- guardar curvas de entrenamiento como datos;
- recalcular el R2 de medias de grupo out-of-sample;
- reencuadrar la memoria como investigacion sobre limites de informacion prepartida;
- dar mas peso a MAE y metricas interpretables;
- preparar explicaciones tecnicas de ICC/R2 y embeddings;
- estructurar la memoria final en 8 paginas + 2 anexos.

## 4. Avance real desde el Informe de Progreso II

El Informe II entro en git en el commit `20df30589`, fechado el 2026-06-01. Desde ese punto hay dos commits en `main`:

- `0856268aa` (2026-06-08): gran cierre metodologico y de artefactos finales.
- `0ffb18433` (2026-06-08): adaptacion para cluster y script Huber.

Cuantitativamente, desde el Informe II hasta `HEAD`:

```text
384 files changed, 44173 insertions(+), 4248 deletions(-)
```

Ademas, a 2026-06-09 hay cambios locales no confirmados:

```text
11 files changed, 442 insertions(+), 27 deletions(-)
```

Y hay nuevos archivos sin trackear relevantes:

- `final/analysis/model_comparison/comparison_secondary_table_residual.csv`
- `final/analysis/model_comparison/residual_context_diagnostics.csv`
- `final/docs/resumen_cambios_recientes_tutor.md`
- `last one/borrador_informe_final.md`
- `last one/resumen_conceptos_tutor.md`

### 4.1 Comparacion principal rehecha bajo protocolo comun

Este es el avance metodologico mas importante desde el Informe II.

Antes, la tabla mezclaba modelos y variantes con informacion adicional: GBT con arquetipos, GBT con Pair TE, MLPs con distintas representaciones. Tras el plan de accion, la tabla principal queda limitada a modelos comparables bajo el mismo protocolo:

```text
10 champion IDs + side
```

Archivo principal:

```text
final/analysis/model_comparison/final_main_table_raw.md
```

Tabla actual:

| Modelo | R2 | Spearman | MAE | within +/-0.20 | seeds |
|---|---:|---:|---:|---:|---:|
| Global Mean | -0.0008 | - | 0.1551 | 0.6889 | 0 |
| Champion Mean | 0.1243 | 0.3362 | 0.1438 | 0.7306 | 0 |
| HistGBT | 0.1595 +/- 0.0004 | 0.3869 +/- 0.0004 | 0.1408 | 0.7420 | 3 |
| MLP OneHot | 0.1536 +/- 0.0010 | 0.3801 +/- 0.0005 | 0.1412 | 0.7397 | 3 |
| MLP Embed Shared | 0.1507 +/- 0.0004 | 0.3763 +/- 0.0007 | 0.1415 | 0.7391 | 3 |
| MLP Per-Role + Interactions | 0.1527 +/- 0.0013 | 0.3783 +/- 0.0014 | 0.1414 | 0.7398 | 3 |

Avance respecto al plan:

- Comparacion justa: completada.
- 3 seeds: completado para modelos aprendidos principales.
- Features comunes: completado.
- `sample_weight`: auditado como usado.
- Variantes enriquecidas movidas a tabla secundaria.

La auditoria esta en:

```text
final/analysis/model_comparison/feature_protocol_audit.csv
```

### 4.2 Correccion de ICC vs R2

Este era un punto critico del tutor. Ahora se separan dos metricas:

- ICC train: descriptivo, in-sample, mide consistencia dentro de grupos.
- R2 group-mean OOS: predictivo, calcula medias en train y las aplica a test.

Archivos:

```text
final/analysis/ceiling/ceiling_methodology_note.md
final/analysis/ceiling/ceiling_oos_summary.csv
final/analysis/ceiling/ceiling_summary.md
```

Resultados actuales:

| Agrupacion | ICC train | R2 group-mean OOS |
|---|---:|---:|
| support_champion | 0.1214 | 0.1249 |
| botlane_champions | 0.1394 | 0.1239 |
| botlane_champions+side | 0.1391 | 0.1132 |
| sup_vs_enemy_sup_champion | 0.1316 | 0.1200 |

Cambio importante respecto al Informe II:

- El 0.173 ya no debe presentarse como referencia predictiva comparable directa.
- La referencia comparable en test para botlane+side es 0.1132.
- HistGBT con protocolo comun alcanza 0.1595, por encima de esa media de grupo OOS.

Esto mejora la defensa: el modelo no solo memoriza medias simples de grupos; capta patrones de contexto del draft que una tabla de medias no generaliza bien.

### 4.3 Experimento residual: senal mas alla del support

Se implemento un diagnostico nuevo para responder si el modelo solo aprende la identidad del support.

Idea:

```text
support_effect = media suavizada del support aliado
residual = y - support_effect
context_model = GBT(resto del draft + interacciones suavizadas)
pred_final = support_effect + context_model
```

Archivos:

```text
final/scripts/25_residual_interaction_experiment.py
final/analysis/model_comparison/residual_context_diagnostics.csv
```

Resultados:

| Modelo | R2 | Spearman | Lectura |
|---|---:|---:|---|
| Smoothed Support Mean | 0.1240 | 0.3356 | efecto base del support |
| Residual Context GBT | 0.0386 sobre residual | 0.1892 sobre residual | senal contextual restante |
| Smoothed Support Mean + Residual Context GBT | 0.1584 | 0.3854 | modelo aditivo diagnostico |

Lift frente a media suavizada del support:

```text
R2: +0.0343
Spearman: +0.0498
```

Interpretacion: el resto del draft si aporta senal adicional, aunque limitada. Esto es muy util para defender que el modelo no se limita a hacer un lookup del support.

Estado: completado como analisis secundario/diagnostico. No debe sustituir la tabla principal.

### 4.4 WandB, curvas e historiales

Los scripts principales de entrenamiento ya soportan `--use-wandb`:

- `final/scripts/03_train_gbt.py`
- `final/scripts/04a_train_mlp_onehot.py`
- `final/scripts/04b_train_mlp_embed.py`
- `final/scripts/04c_train_mlp_per_role.py`
- `final/scripts/04d_train_mlp_per_role_huber.py`
- `final/scripts/run_all_training.py`

Tambien existen runs locales de WandB fechados el 2026-06-03 en `wandb/`.

Las MLPs guardan historiales como CSV:

- `final/models/mlp_onehot/history.csv`
- `final/models/mlp_embed/history.csv`
- `final/models/mlp_per_role/history.csv`
- `final/models/mlp_per_role_huber/history.csv` si se ejecuta el script nuevo
- historiales del grid de HP search en `final/analysis/hp_search/runs/*/history.csv`

Tambien hay curvas graficas en:

```text
final/analysis/training_curves/
```

Estado: funcionalmente completado. Pendiente: decidir que curvas finales se muestran en memoria y evitar usar graficas antiguas que no correspondan al protocolo final.

### 4.5 Huber loss y adaptacion a cluster

Se anadio:

```text
final/scripts/04d_train_mlp_per_role_huber.py
submit_mlp_huber.sh
requirements-cluster.txt
```

Objetivo:

- entrenar una variante MLP Per-Role con Huber/SmoothL1 para reducir sensibilidad a errores grandes;
- combinar robustez de loss con `sample_weight`;
- facilitar ejecucion en cluster.

Estado: implementado como experimento adicional. Todavia parece secundario frente al cierre metodologico principal. Conviene decidir si se incluye en memoria como prueba complementaria o se deja fuera para no abrir demasiado el foco.

### 4.6 Analisis de embeddings mejorado

Desde el Informe II se reforzo el analisis de embeddings:

- comparaciones de embeddings shared vs per-role;
- vecinos cercanos;
- distancias vs diferencia de roam score;
- proyecciones y graficas;
- tests especificos para vecinos.

Archivos relevantes:

```text
final/scripts/17_embedding_analysis.py
final/analysis/embedding_analysis/
tests/test_embedding_analysis_neighbors.py
last one/resumen_conceptos_tutor.md
```

Estado: mejorado para explicacion, no para cambiar la conclusion de resultados. La conclusion sigue siendo: embeddings aprenden algo, pero no superan a HistGBT.

### 4.7 Auditorias de geometria y posiciones invalidas

Se anadieron scripts y artefactos para revisar exclusiones de posiciones:

```text
final/scripts/23_plot_invalid_frame_positions.py
final/scripts/24_plot_frame_calculation_exclusions.py
final/analysis/invalid_frame_positions/
```

Esto refuerza la parte de etiqueta/geometria, mostrando que se auditaron casos excluidos o posiciones invalidas.

Estado: avance util para anexos o defensa metodologica. No deberia ocupar espacio central salvo que el tutor pregunte por fiabilidad de la etiqueta.

### 4.8 Borrador y narrativa final

Se crearon nuevos documentos:

```text
last one/borrador_informe_final.md
last one/resumen_conceptos_tutor.md
final/docs/resumen_cambios_recientes_tutor.md
last one/estructura_informe_final.md
last one/explicacion_tecnica_completa.md
```

La narrativa ya esta reencuadrada hacia:

> Hasta que punto la configuracion prepartida de un sistema multiagente permite anticipar la movilidad temprana de un agente funcional de apoyo.

Esto resuelve parcialmente el problema de vocabulario demasiado LoL. La memoria final puede mantener League of Legends como caso de estudio, pero debe hablar primero en terminos generales:

- campeon -> agente;
- draft -> configuracion prepartida;
- support -> agente de apoyo;
- ADC -> companero principal de zona / tirador;
- botlane -> zona inferior / zona asignada inicial;
- roaming -> movilidad temprana fuera de zona asignada;
- partidas caoticas -> ejecucion temprana anomala.

Estado: borrador avanzado, pendiente de pulido formal.

## 5. Estado actual frente al checklist del plan de accion

| Punto del plan | Estado | Evidencia |
|---|---|---|
| Modelos en igualdad de condiciones | Hecho | `final_main_table_raw.md`, `feature_protocol_audit.csv` |
| Quitar summoner spells de tabla principal | Hecho | protocolo `draft_10_champions_side` |
| `sample_weight` consistente | Hecho para modelos principales | auditoria de protocolo |
| 3 seeds y media +/- std | Hecho en modelos aprendidos principales | tabla final, `n_seeds=3` |
| WandB integrado | Hecho funcional | `--use-wandb`, carpeta `wandb/` |
| Curvas/historiales como datos | Hecho en MLPs y HP search | `history.csv` |
| Ceiling R2 out-of-sample | Hecho | `ceiling_oos_summary.csv` |
| ICC explicado como metrica distinta | Hecho | `ceiling_methodology_note.md` |
| Tabla comparativa final justa | Hecho | `final_main_table_raw.md` |
| Analisis residual de senal contextual | Hecho adicional | `residual_context_diagnostics.csv` |
| Borrador de informe con vocabulario general | En progreso avanzado | `borrador_informe_final.md` |
| Explicacion tecnica de embeddings | Hecho para defensa | `resumen_conceptos_tutor.md`, `explicacion_tecnica_completa.md` |
| Prototipo CLI pulido | Parcial / pendiente de validacion final | `predict_cli.py` modificado antes; no verificado en esta recopilacion |
| Tests ejecutados en esta revision | Pendiente | `pytest` no esta instalado en `.venv` ni en el Python bundled |
| Limpieza de repo y commit final | Pendiente | hay cambios locales y archivos sin trackear |

## 6. Cuanto has avanzado desde el Informe II

Mi estimacion cualitativa:

- Avance tecnico-metodologico desde Informe II: alto. Los puntos que el tutor podia atacar con mas fuerza ya tienen respuesta: protocolo justo, seeds, sample weights, ICC/R2 separado, diagnostico residual, documentacion de embeddings y tabla final actualizada.
- Avance experimental: muy alto para la tarea support. El proyecto ya no depende de una sola tabla del Informe II; ahora tiene auditorias, referencias OOS, secundarios y diagnosticos.
- Avance de memoria final: medio-alto. Ya hay borrador y reencuadre conceptual, pero falta convertirlo en documento final compacto, coherente y con figuras definitivas.
- Avance de entrega completa: todavia no cerrado. Quedan limpieza de repo, decision sobre que secundarios incluir, verificacion de tests, pulido CLI y preparacion de defensa.

En terminos practicos: desde el Informe II no solo se ha "avanzado un poco"; se ha corregido la base metodologica del proyecto. El TFG ahora esta mucho mas defendible que el 1 de junio. Lo pendiente ya no parece investigacion abierta principal, sino consolidacion: elegir la narrativa final, limpiar artefactos, verificar reproducibilidad y preparar la defensa.

## 7. Mensaje central actualizado

La tesis final del trabajo deberia formularse asi:

> Este TFG estudia hasta que punto la configuracion prepartida de un sistema multiagente permite anticipar la movilidad temprana de un agente de apoyo. A partir de 383.247 observaciones, se construye una etiqueta espacial continua validada frente a conocimiento experto. Los modelos muestran que existe senal predictiva en el draft: el support por si solo explica una parte relevante, y el draft completo aporta contexto adicional. Sin embargo, la mayor parte de la varianza queda fuera del alcance prepartida porque depende de la ejecucion concreta, eventos tempranos y comportamiento de jugadores. El resultado principal no es un predictor perfecto, sino la cuantificacion y defensa de ese limite.

## 8. Pendientes recomendados antes de cerrar

Prioridad alta:

- Dejar limpia la tabla final que ira a memoria: probablemente `HistGBT`, `MLP Per-Role + Interactions`, `Champion Mean`, `Global Mean` y metricas practicas.
- Actualizar cualquier texto que aun diga que el ceiling comparable es 0.173. Debe distinguirse ICC/train/in-sample de R2 OOS.
- Decidir si el experimento residual va en resultados principales, discusion o anexo.
- Verificar que `predict_cli.py` sigue funcionando con el modelo final y el protocolo actual.
- Instalar/usar un entorno con `pytest` y ejecutar tests antes de entrega.
- Confirmar que `final/models/` esta correctamente ignorado y que no se va a commitear peso de modelos accidentalmente.

Prioridad media:

- Seleccionar 2-3 figuras definitivas: geometria/heatmap, tabla principal, clean vs chaotic, talvez diagrama de pipeline.
- Preparar explicacion oral de embeddings, ICC/R2, predicciones comprimidas y partidas caoticas.
- Revisar si Huber se incluye como experimento secundario o se omite para no dispersar el relato.
- Limpiar artefactos temporales y dejar en `last one` solo documentos utiles para memoria/tutor.

Verificacion pendiente de esta recopilacion:

- Intente ejecutar tests con `.venv/Scripts/python.exe -m pytest` y con el Python bundled de Codex, pero ambos entornos devuelven `No module named pytest`. No he instalado dependencias ni he modificado entornos.
