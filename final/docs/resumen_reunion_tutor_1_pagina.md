# Resumen para reunion con tutor

## Mensaje principal

El TFG ha evolucionado desde una propuesta amplia de clasificacion multi-output hacia un caso de estudio mas acotado y defendible: estimar la tendencia de roaming del support a partir del draft. El resultado no debe leerse como una prediccion exacta de cada partida, sino como una medicion de cuanta senal pre-partida contiene la composicion de campeones.

## Que se decidio y por que

| Decision | Justificacion |
| --- | --- |
| Pasar de clasificacion a regresion continua | El comportamiento no se separaba bien en clases; la clase intermedia era ambigua y se perdia informacion al discretizar scores continuos. |
| Centrarse en support-only | Permite cerrar una tarea con rigor: etiqueta, baselines, modelos, techo empirico, errores, limitaciones y prototipo. Jungla/equipo quedan como extension futura. |
| Separar draft y timeline | El draft es input pre-partida; la timeline solo construye la etiqueta. Asi se evita data leakage y se mantiene la pregunta original. |
| Usar geometria manual del mapa | La etiqueta depende de distinguir contexto de botlane frente a zonas compatibles con roaming; la geometria manual es mas interpretable. |
| Comparar con baselines y techo empirico | Un R2 aislado no era interpretable. Las baselines permiten saber si el modelo aprende algo mas que reglas triviales. |

## Resultados clave

| Resultado | Valor |
| --- | ---: |
| Dataset final | 383.247 observaciones partida-equipo |
| Comparacion experta por campeon | Spearman 0.825 |
| Primera MLP del Informe I | R2 0.13068, Spearman 0.3568 |
| Mejor modelo final | HistGBT + Pair Target Encoding |
| Mejor resultado final | R2 0.161, Spearman 0.388, MAE 0.141 |
| Baseline media por campeon support | R2 0.125 |
| Referencia empirica botlane+lado | R2 0.173 |
| Predicciones dentro de ±0.20 | 74.2% |
| Partidas caoticas | ~26.5% de observaciones |

## Interpretacion

- El draft contiene senal predictiva real, pero limitada.
- El campeon support explica gran parte de la senal.
- El resto del draft aporta informacion adicional, aunque con margen moderado.
- El mejor modelo queda cerca de la referencia empirica por botlane+lado.
- Las MLPs y embeddings no superan al modelo tabular.
- Los errores grandes suelen estar asociados a partidas caoticas, donde la separacion support-ADC no refleja necesariamente una intencion de roaming.

## Conclusion para defender

El proyecto no demuestra que el draft determine el comportamiento del support. Demuestra algo mas prudente: el draft permite estimar una predisposicion parcial al roaming, coherente con conocimiento experto y superior a baselines simples, pero la ejecucion concreta depende de factores que no estan disponibles antes de empezar la partida.

## Preguntas al tutor

1. Confirmar si el alcance support-only es adecuado para el cierre del TFG.
2. Decidir como presentar el R2: resultado modesto, pero contextualizado por baselines y referencia empirica.
3. Elegir si en la memoria se enfatiza mas la construccion de la etiqueta o la comparacion de modelos.
4. Decidir si el prototipo CLI se muestra en defensa como cierre aplicado.

