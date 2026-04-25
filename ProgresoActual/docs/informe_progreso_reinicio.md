# Informe de progreso: reinicio metodologico del TFG

## 1. Propuesta inicial y objetivo original

El objetivo inicial del TFG era inferir tendencias tempranas en partidas de
League of Legends a partir de informacion previa al inicio de la partida,
principalmente el draft. La hipotesis de trabajo era que ciertos patrones de
composicion podian anticipar comportamientos tempranos como presencia del
jungla, roaming del support o tendencia lateral del equipo.

La primera formulacion se planteo como clasificacion. Para ello se definieron
scores continuos a partir de eventos y posiciones observadas durante los
primeros minutos, y despues esos scores se discretizaron en clases. Esta
aproximacion permitio construir un pipeline completo, pero tambien introdujo una
decision metodologica problematica: convertir una senal continua y subjetiva en
clases rigidas antes de entrenar el modelo.

## 2. Resumen del pipeline existente

El repositorio contiene una primera arquitectura completa:

- Recoleccion de partidas y timelines desde la API de Riot.
- Procesado de raw JSON para construir features de draft y etiquetas.
- Entrenamiento de modelos multi-output con PyTorch.
- Analisis exploratorio, figuras, estudios de estabilidad y comparaciones entre
  configuraciones.

El trabajo previo no se descarta. Sirve como historial experimental y como base
para justificar el cambio de direccion. Sin embargo, el nuevo avance se centra
en una version mas pequena: support-only, score continuo, regresion y tracking
experimental.

## 3. Fase 1: ventanas temporales y estabilidad

La Fase 1 estudio si la ventana temporal usada para construir las etiquetas
condicionaba el rendimiento. Se compararon ventanas de early game y se observo
que las tres tareas no se estabilizaban al mismo ritmo. Jungle tendia a
estabilizarse antes, support necesitaba algo mas de tiempo y team tendency era
la tarea mas inestable.

La conclusion principal fue que la definicion del target era mas importante que
la arquitectura del modelo. El problema no era solo que el modelo aprendiera
poco, sino que la etiqueta mezclaba intencion estrategica, ejecucion real y
ruido temprano de partida.

## 4. Fase 2: problema de `ambiguous`

La Fase 2 analizo el papel de la clase intermedia `ambiguous`. Los experimentos
compararon formulaciones ternarias y binarias. El resultado fue claro: eliminar
o reducir la clase intermedia mejoraba las metricas, especialmente cuando se
conservaban solo ejemplos extremos.

Este resultado fue util, pero tambien mostro la limitacion de la formulacion de
clasificacion. El score original contenia informacion gradual, y convertirlo en
clases obligaba a tomar decisiones arbitrarias sobre umbrales y zonas centrales.

## 5. Giro hacia regresion continua

Tras revisar el planteamiento con el profesor, el nuevo enfoque pasa a predecir
directamente un score continuo en `[0, 1]`. La primera tarea elegida es support,
porque es mas concreta, facil de inspeccionar por campeon y relativamente
defendible desde el punto de vista experto.

El nuevo pipeline separa dos costes:

- Extraccion cara: leer raw JSON y construir `support_frame_state`.
- Experimentacion barata: recalcular scores desde parquet con distintas
  heuristicas sin volver a leer las partidas.

Esta separacion permite iterar sobre la formulacion de la etiqueta en minutos,
no en horas.

## 6. Nueva etiqueta de support

La etiqueta continua de support se calcula como una combinacion ponderada de
senales observables:

- ratio de frames fuera de la zona extendida de bot;
- ratio de frames lejos del ADC;
- penalizacion basada en diferencia relativa de experiencia con el ADC.

La implementacion permite variar ventana, minuto de inicio, umbral de distancia
y pesos de la formula. Cada configuracion queda trazada con `config_id` y puede
exportarse como `support_scores.parquet` para entrenamiento.

En el primer baseline completo `m12`, la distribucion observada queda desplazada
hacia valores bajos, pero no colapsada. Esto es coherente con la interpretacion
del score: no todos los supports abandonan la linea de forma intensa en todas las
partidas. Tambien aparece una ligera asimetria por lado, con media superior en
blue side. Esta diferencia se interpreta junto a un hallazgo temprano del
proyecto: el equipo azul tomaba las larvas primero aproximadamente en el `60%`
de las partidas analizadas. Dado que los accesos a dragon, larvas, heraldo y
baron no son perfectamente simetricos, parte de la asimetria del roaming puede
ser estructura real del mapa y del metajuego, no necesariamente ruido.

La referencia por campeon usada hasta ahora combina la idea de metadatos
oficiales con una tabla experta manual. En la practica actual, el score experto
`expert_support_roam_score` procede de
`references/manual_support_champion_reference.csv`: una curacion subjetiva por
campeon con arquetipo, score esperado `[0,1]`, confianza y notas. Data Dragon no
proporciona un score oficial de roaming; solo sirve como posible fuente de tags y
metadatos generales.

## 7. Decisiones acertadas y equivocadas

Decisiones acertadas:

- Construir un pipeline completo desde raw hasta entrenamiento.
- Guardar artefactos, metricas y figuras para poder auditar el avance.
- Analizar estabilidad temporal antes de seguir aumentando complejidad.
- Detectar que `ambiguous` era una fuente importante de ruido.

Decisiones equivocadas o mejorables:

- Empezar por clasificacion antes de validar si el target debia ser continuo.
- Tratar las tres tareas como si fueran equivalentes.
- Reentrenar y regenerar etiquetas desde raw para cada variante, ralentizando la
  iteracion.
- Dedicar esfuerzo a hiperparametros antes de cerrar una definicion solida de la
  etiqueta.

## 8. Estado actual

El reinicio implementa:

- scorer rapido de support desde `support_frame_state`;
- exportacion de configuraciones de score compatibles con el model input;
- trainer MLP support-only con `OneHotEncoder`, `MSELoss` y metricas de
  regresion;
- integracion opcional con Weights & Biases;
- referencia manual/oficial de campeones y comparacion por campeon;
- este informe editable como base de seguimiento.

Los artefactos nuevos del reinicio deben permanecer dentro de `ProgresoActual/`.
Las carpetas heredadas (`data/`, `data_new/` y `PropuestaInicial/`) pueden actuar
como fuentes de entrada o archivo historico, pero no como destino por defecto de
experimentos nuevos.

## 9. Planificacion de 8 semanas

| Semana | Tarea | Descripcion | Entregable esperado | Estado | Motivo si no se cumple |
|---|---|---|---|---|---|
| 1 | Limpieza conceptual y benchmark | Separar propuesta inicial, progreso nuevo y medir tiempos actuales. | README, informe base y tiempo de scoring en sample5. | Pendiente | - |
| 2 | Scorer rapido de support | Usar `support_frame_state` para probar heuristicas sin releer raw. | Grid de configs y `support_scores` elegido. | Pendiente | - |
| 3 | Trainer support-only | Entrenar MLP con target continuo y metricas de regresion. | Modelo base, `metrics.json`, `history.csv`. | Pendiente | - |
| 4 | W&B y comparacion de heuristicas | Registrar curvas, configs y metricas por heuristica. | Runs offline/online comparables. | Pendiente | - |
| 5 | Referencia de campeones | Combinar Data Dragon con tabla experta manual. | CSV de referencia y analisis por campeon. | Pendiente | - |
| 6 | Seleccion de etiqueta candidata | Elegir formula final segun metricas, distribucion y coherencia experta. | Config final justificada. | Pendiente | - |
| 7 | Redaccion de resultados | Convertir experimentos en figuras y texto academico. | Seccion de metodologia/resultados. | Pendiente | - |
| 8 | Revision final | Revisar conclusiones, amenazas a validez y entrega. | Version lista para tutor. | Pendiente | - |

## 10. Proximos pasos inmediatos

1. Ejecutar smoke test sobre `sample5`.
2. Comparar varias heuristicas de support y seleccionar una primera candidata.
3. Entrenar MLP support-only con W&B offline.
4. Revisar desviaciones por campeon contra la tabla experta.
5. Actualizar este informe con metricas reales.

## 11. Lineas futuras

Una posible ampliacion metodologica, sugerida durante la tutorizacion, es
explorar arquitecturas secuenciales como RNN, GRU o LSTM. La motivacion es que
la timeline de la API de Riot no es solo una tabla agregada, sino una secuencia
de "fotografias" del estado de la partida. El enfoque actual con MLP resume esa
informacion en una etiqueta continua y entrena desde features de draft; una
arquitectura secuencial permitiria modelar directamente la evolucion temporal de
posiciones, experiencia, distancia al ADC y presencia en zonas del mapa.

Esta linea queda fuera del primer reinicio porque antes se necesita una baseline
tabular solida y defendible. Si la MLP aprende senal real y la etiqueta queda
validada, una RNN podria plantearse como comparacion futura para comprobar si
usar la secuencia completa mejora la prediccion o ayuda a explicar mejor los
patrones tempranos de roaming.
