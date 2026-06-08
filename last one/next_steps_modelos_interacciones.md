# Next steps para mejorar modelos con interacciones de draft

## Contexto

Los resultados actuales sugieren que el campeon support aliado explica una parte importante del `support_roam_score`. Esto no es necesariamente un problema: es una senal real del dominio. Sin embargo, tambien puede hacer que los modelos aprendan una solucion conservadora basada en la media del support y no exploten suficientemente interacciones mas finas del draft, como:

- Bard con Ezreal frente a Bard con Kog'Maw.
- Thresh contra Leona frente a Thresh contra Soraka.
- Nautilus con junglas de gank frente a junglas de farming.

El experimento residual ya implementado apunta a que existe senal contextual mas alla de la media del support:

```text
Support Mean Baseline:
R2 = 0.1215
Spearman = 0.3332

Support Mean + Residual Interaction GBT:
R2 = 0.1537
Spearman = 0.3793
Lift R2 sobre support mean = +0.0321
```

Esto no sustituye al modelo principal, pero justifica probar modelos globales que incorporen interacciones explicitas de forma controlada.

## 1. HistGBT global con features de interaccion suavizadas

### Objetivo

Entrenar un unico modelo global que prediga directamente `support_roam_score`, pero enriqueciendo la entrada con variables numericas que representen interacciones relevantes de draft.

La idea es pasar de:

```text
HistGBT(10 campeones + side)
```

a:

```text
HistGBT(
  10 campeones + side
  + target encodings suavizados de interacciones
)
```

### Interacciones candidatas

Usaria las mismas familias que en el experimento residual:

- `ally_utility_champion_id + ally_bottom_champion_id`
  - Sinergia support + ADC.
- `ally_utility_champion_id + enemy_utility_champion_id`
  - Matchup support aliado contra support enemigo.
- `ally_utility_champion_id + ally_jungle_champion_id`
  - Potencial de setup/gank con jungla.
- `ally_utility_champion_id + ally_middle_champion_id`
  - Potencial de roam hacia mid.
- `ally_utility_champion_id + ally_bottom_champion_id + enemy_utility_champion_id`
  - Contexto de botlane simplificado.

### Por que puede ayudar

Los arboles pueden quedarse en splits fuertes y faciles, por ejemplo la identidad del support aliado. Si se les da una variable numerica ya suavizada como "efecto historico de Bard+Ezreal", el modelo puede usar esa informacion de interaccion sin tener que descubrirla mediante muchos splits condicionales.

Esto mantiene la ventaja del modelo principal:

```text
score = f(draft completo)
```

pero le facilita senal de segundo orden.

### Validacion necesaria

Es imprescindible evitar leakage:

- En train, los target encodings deben ser out-of-fold.
- En val/test, los mapas de encoding deben ajustarse solo con train.
- Las combinaciones raras deben suavizarse hacia la media global.
- Debe guardarse el mapping y el config del protocolo usado.

Metricas a comparar:

- R2.
- Spearman.
- MAE.
- `pred_std / target_std`.
- Mejora frente a HistGBT principal y Champion Mean.

### Criterio de exito

Seria candidato a modelo mejorado si supera al HistGBT principal en test o, al menos, mejora Spearman/R2 sin empeorar claramente MAE.

## 2. Mini tuning del HistGBT con interacciones

### Objetivo

Probar si el modelo enriquecido necesita mas flexibilidad para capturar interacciones, sin abrir una busqueda enorme de hiperparametros.

### Configuraciones propuestas

Probaria una cuadricula pequena:

```text
max_depth: 6, 8
min_samples_leaf: 20, 50
learning_rate: 0.02, 0.05
l2_regularization: 0, 1, 5
```

Opcionalmente, si el entrenamiento sigue siendo rapido:

```text
max_iter: 300, 600
```

### Por que puede ayudar

Las interacciones de 2 o 3 variables pueden requerir arboles algo mas profundos. Reducir `min_samples_leaf` permite capturar subgrupos mas especificos, mientras que `l2_regularization` ayuda a no memorizar ruido.

### Riesgo

Es facil sobreajustar. Por eso no conviene hacer una busqueda grande ni elegir el mejor modelo solo por validation sin una evaluacion final estricta.

### Validacion necesaria

Para cada configuracion:

- Entrenar en train.
- Seleccionar por validation.
- Evaluar una unica vez en test el mejor candidato.
- Guardar tabla con todos los intentos, no solo el ganador.

Tambien conviene mirar:

```text
train R2 vs val R2
pred_std / target_std
MAE
```

Si sube mucho train y no sube val/test, esta memorizando.

## 3. Calibracion post-hoc de dispersion

### Objetivo

Corregir la compresion de predicciones sin cambiar el modelo base.

Muchos modelos regresores predicen cerca de la media porque el target es ruidoso. Una calibracion simple puede aumentar la dispersion:

```text
pred_calibrada = media_train + alpha * (pred_original - media_train)
```

Si `alpha > 1`, las predicciones se alejan de la media.

### Por que puede ayudar

Si el modelo ordena relativamente bien los drafts, pero sus scores estan infra-dispersos, esta calibracion puede mejorar R2 y acercar `pred_std` a `target_std`.

Importante: no cambia el ranking si `alpha` es positivo. Por tanto, Spearman deberia mantenerse casi igual. Lo que cambia es la escala.

### Como validarlo

Procedimiento defendible:

1. Entrenar el modelo normalmente en train.
2. Generar predicciones en validation.
3. Buscar `alpha` en validation, por ejemplo:

```text
alpha: 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5
```

4. Fijar el mejor `alpha`.
5. Evaluar una unica vez en test.

Metricas clave:

- R2.
- MAE.
- Spearman.
- `pred_std / target_std`.
- Porcentaje de predicciones clipeadas a `[0, 1]`.

### Riesgo

Puede mejorar R2 pero empeorar MAE si exagera demasiado los extremos. Tambien puede crear scores mas vistosos sin aportar informacion real. Por eso debe tratarse como calibracion, no como nuevo aprendizaje.

### Criterio de exito

La calibracion seria util si:

- mejora R2 en test;
- no empeora MAE de forma relevante;
- aumenta `pred_std / target_std`;
- mantiene Spearman.

## 4. Ablacion con loss `absolute_error`

### Objetivo

Comprobar si una perdida mas robusta al ruido produce predicciones mas utiles que la perdida cuadratica por defecto.

En HistGBT se puede probar:

```text
loss = "squared_error"
loss = "absolute_error"
```

En MLP, una version equivalente seria comparar:

```text
weighted MSE
weighted L1
weighted Huber / SmoothL1
```

### Por que puede ayudar

MSE penaliza mucho los errores grandes. En un target ruidoso como roaming pregame, algunas partidas pueden ser impredecibles desde draft. MSE puede responder a ese ruido generando predicciones conservadoras.

MAE optimiza una tendencia mas cercana a la mediana condicional. Puede ser mas robusta si hay outliers o partidas caoticas.

### Matiz importante

MAE o Huber no garantizan mas varianza de prediccion. Pueden mejorar MAE, pero tambien seguir produciendo predicciones comprimidas si la senal disponible es debil.

Por eso lo trataria como ablacion pequena, no como apuesta principal.

### Validacion necesaria

Comparar:

```text
HistGBT main squared_error
HistGBT main absolute_error
HistGBT interactions squared_error
HistGBT interactions absolute_error
```

Metricas:

- R2.
- MAE.
- Spearman.
- `pred_std / target_std`.

### Criterio de exito

Seria interesante si `absolute_error` reduce MAE sin destruir R2/Spearman, o si mejora la dispersion sin empeorar la calidad general.

## Orden recomendado

1. Implementar y evaluar `HistGBT + smoothed interaction features`.
2. Hacer mini tuning solo sobre ese modelo enriquecido.
3. Aplicar calibracion de dispersion al mejor candidato.
4. Probar `absolute_error` como ablacion pequena.

## Que no priorizaria ahora

No priorizaria DeepFM ni embeddings preentrenados en esta fase.

Son ideas teoricamente buenas, pero abririan demasiada superficie metodologica:

- nueva arquitectura;
- nuevos hiperparametros;
- explicacion mas compleja;
- mayor riesgo de errores de implementacion;
- dificil comparacion justa con el pipeline actual.

Para el TFG, es mas defendible agotar primero la linea HistGBT enriquecida, porque conecta directamente con el modelo principal y con el experimento residual ya implementado.

## Lectura posible para el informe

Si el modelo enriquecido mejora:

> La introduccion de interacciones suavizadas permite al modelo aprovechar senal contextual que quedaba parcialmente eclipsada por la identidad del support aliado.

Si no mejora:

> Aunque el analisis residual detecta cierta senal contextual, esta no se traduce en una mejora robusta frente al modelo principal en evaluacion hold-out. Esto sugiere que gran parte de la informacion predictiva disponible en pregame ya esta capturada por la identidad del support y que el resto de variacion depende de eventos no observados de partida.

Ambas conclusiones son defendibles.
