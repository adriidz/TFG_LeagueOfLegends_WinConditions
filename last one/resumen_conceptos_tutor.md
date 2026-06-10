# Guía Conceptual y Técnica para Reunión con Tutor

Este documento resume de forma intuitiva, pero con el rigor necesario, los cuatro puntos críticos del TFG. Cada sección empieza con una **explicación conceptual** (para entender el "qué" y el "por qué"), seguida de la **fórmula clave simplificada**, y termina con su **valor real en el proyecto** para defenderlo ante tu tutor.

---

## 1️⃣ El Techo Empírico: Train ICC vs. $R^2$ Out-of-Sample (OOS)

> [!NOTE]
> **Pregunta del tutor:** *"¿Cómo es que de un ICC sacáis un R²?"*
> **Tu respuesta clave:** No se saca uno de otro. El **ICC** es un análisis estadístico para describir la consistencia del juego in-sample. El **$R^2$ de grupo** es una simulación predictiva out-of-sample. Son dos formas independientes de medir lo mismo sobre los mismos grupos.

### A. Explicación Conceptual
Imagina que agrupamos todas las partidas que tienen la misma pareja de botlane (ej. Thresh + Ezreal). Queremos saber: **¿El score de roaming es siempre parecido para Thresh+Ezreal, o varía una barbaridad de una partida a otra?**

*   **El ICC (Coeficiente de Correlación Intraclase):** Es una métrica que se calcula **utilizando un ANOVA (Análisis de Varianza)**. El ANOVA realiza el trabajo de dividir la varianza total en dos cajones:
    1.  La varianza **entre grupos** (cuánto cambia el roaming al cambiar de campeones).
    2.  La varianza **dentro de los grupos** (cuánto cambia el roaming entre diferentes partidas jugando la misma botlane, debido a que el support muere rápido, el ADC se queda solo, etc.).
    El **ICC** es simplemente el número final (entre 0 y 1) que resume esta división: nos dice qué porcentaje de la varianza total corresponde al primer cajón (el del draft). Si el ICC es cercano a 0, significa que jugar Thresh+Ezreal no garantiza nada porque la variabilidad entre partidas individuales es gigante. Si es cercano a 1, el roaming está predeterminado por el draft.
*   **El $R^2$ Group-Mean OOS (Out-of-Sample):** Aquí hacemos de "modelos". Memorizamos la media de roaming de Thresh+Ezreal usando solo los datos de **entrenamiento** (train). Luego, vamos al conjunto de **test** y predecimos esa media para las partidas de Thresh+Ezreal. Medimos el error de predicción ($R^2$). Esto es un techo predictivo real y justo porque evalúa datos no vistos.

---

### B. Ecuación Clave
El ICC calcula los cuadrados medios entre grupos ($MS_B$, diferencias entre composiciones) y dentro de los grupos ($MS_W$, diferencias entre partidas con la misma composición):

$$ICC = \frac{MS_B - MS_W}{MS_B + (\bar{n} - 1)MS_W}$$

*(Donde $\bar{n}$ es simplemente el promedio de partidas por grupo de draft).*

El $R^2$ predictivo mide cuánto mejor predice nuestra tabla de medias comparada con el promedio global:

$$R^2 = 1 - \frac{\text{Suma del error del modelo}^2}{\text{Suma del error del promedio global}^2}$$

---

### C. Valor en el Proyecto
*   **¿Para qué sirve?** El ICC de botlane es de **0.139** (indica que el 13.9% del roaming se debe a la composición estable). El $R^2$ predictivo OOS de botlane+side es de **0.1132**. 
*   Esto nos dice que una tabla de lookup memorizada tiene un límite predictivo de ~11%. 
*   Como nuestro modelo **HistGBT alcanza un $R^2$ de 0.1595** (superando al techo de la botlane), demostramos que el modelo no solo memoriza el support o la botlane, sino que **aprende interacciones adicionales de los otros 8 slots del draft** que una media simple no puede capturar.

---

## 2️⃣ Rigor e Igualdad de Condiciones

> [!NOTE]
> **Queja del tutor:** *"No es justo comparar HistGBT con arquetipos contra una MLP normal, y no se ven todas las curvas."*
> **Tu respuesta clave:** Toda la comparación principal se ha rehecho bajo un **único protocolo común**. Todos los modelos de la tabla principal se entrenan exactamente con el mismo split, las mismas variables (10 campeones + lado) y las mismas semillas.

### A. Explicación Conceptual
Para que una carrera sea justa, todos los atletas deben correr la misma distancia y con el mismo calzado. Antes, el HistGBT usaba variables extra creadas a mano (como arquetipos o codificaciones de parejas), mientras que la MLP usaba solo los campeones individuales.

Ahora se ha establecido un **Protocolo de Entrada Común**:
*   Todos los modelos reciben únicamente: **10 Champion IDs + Lado (blue/red)**.
*   Todos los modelos neuronal/árboles se entrenan usando las mismas 3 semillas aleatorias para reportar medias y desviaciones estándar estables.
*   Las curvas de entrenamiento ahora son auditables porque se guardan numéricamente en un historial por época (`history.csv`) y se grafican limpiamente.

---

### B. Tabla Comparativa (Métricas Reales en Test)
*Los modelos aprendidos se evalúan de forma idéntica ([final_main_table_raw.md](file:///c:/Users/adria/Desktop/TFG/final/analysis/model_comparison/final_main_table_raw.md)):*

| Modelo | R² | Spearman | MAE (Error medio) | pred_std (Dispersión) |
| :--- | :---: | :---: | :---: | :---: |
| **Global Mean** | -0.0008 | — | 15.51% | 0.00% |
| **Champion Mean** | 0.1243 | 0.3362 | 14.38% | 6.80% |
| **HistGBT** | **0.1595 ± 0.0004** | **0.3877 ± 0.0004** | **14.08%** | 7.40% |
| **MLP Per-Role + Inter.** | 0.1527 ± 0.0013 | 0.3783 ± 0.0012 | 14.14% | 7.65% |

---

### C. Valor en el Proyecto
*   **La cercanía de los resultados:** Verás que todos los modelos aprendidos dan resultados muy juntos (R² entre 15% y 16%). Esto **no es un fallo**, es un **hallazgo**: la señal prepartida tiene un límite físico insalvable. El resto es la ejecución de los jugadores durante el transcurso de la partida.
*   **La compresión de predicciones:** La desviación del target real es de ~19% (`target_std`), pero el modelo predice con una desviación de ~7.4% (`pred_std`). Esto conceptualmente significa que el modelo **ordena bien** las composiciones (por eso el Spearman es alto, 0.38), pero **evita predecir extremos** porque sabe que hay demasiado ruido aleatorio in-game.

---

## 3️⃣ HistGBT vs. MLP (Estructura de Modelos)

> [!NOTE]
> **Pregunta del tutor:** *"¿Qué diferencia hay entre HistGBT y una MLP? ¿Por qué uno funciona mejor?"*
> **Tu respuesta clave:** HistGBT es un conjunto de árboles que segmenta el draft mediante preguntas binarias sobre los IDs de los campeones. La MLP es una red continua que multiplica vectores numéricos y requiere mapear categorías discretas a espacios continuos.

### A. Explicación Conceptual
*   **HistGradientBoostingRegressor (GBT):** Funciona como el juego de *"Quién es quién"*. Para procesar campeones categóricos, calcula el roaming promedio de cada campeón y los ordena. Luego, el árbol hace preguntas del tipo: *¿El support aliado está en el grupo de los que más roam hacen (ej. Bard, Pyke, Thresh)?* Si sí, va por la izquierda; si no, por la derecha. Es muy rápido y excelente para capturar interacciones directas.
*   **MLP One-Hot (Red Neuronal sin compresión):** Trata a cada campeón como un interruptor binario independiente (vector con 172 ceros y un 1). Para la red, Thresh y Pyke (ambos roamers) están a la misma distancia que Thresh y Yuumi (opuestos). Esto obliga a la primera capa a tener miles de conexiones ($332K$ parámetros), lo que provoca **overfitting inmediato** (se aprende de memoria el ruido a las pocas épocas).

---

### B. Ecuación Conceptual de MLP
En One-Hot, la red calcula la salida de la primera capa sumando directamente las columnas de la matriz de pesos $W$ correspondientes a los 10 campeones activos:

$$z = W_{champion\_1} + W_{champion\_2} + ... + W_{champion\_10} + \text{lado} \cdot W_{side} + b$$

Con tantos pesos libres y drafts dispersos, la red memoriza fácilmente combinaciones raras en lugar de generalizar.

---

### C. Valor en el Proyecto
El HistGBT sigue siendo ligeramente superior (R² = 0.1595 vs 0.1536 de MLP One-Hot) porque su estructura jerárquica de árbol se adapta de forma natural a la naturaleza discreta del draft, mientras que las redes neuronales necesitan estructurar mejor sus entradas para no sobreajustar.

---

## 4️⃣ Embeddings Categóricos: ¿Cómo aprende la red?

> [!NOTE]
> **Pregunta del tutor:** *"¿Cómo justifica que embeddings no es One-Hot y cómo se modifican al aprender?"*
> **Tu respuesta clave:** Un embedding es una **tabla de coordenadas entrenable** (matriz $173 \times 16$). En lugar de un vector gigante one-hot, cada campeón se representa con 16 números continuos. Estos números cambian mediante **backpropagation** en cada batch de entrenamiento para reducir el error de roaming.

### A. Explicación Conceptual
Imagina que le damos a cada uno de los 173 campeones una "tarjeta de coordenadas" con 16 valores inicialmente aleatorios.
*   **Inicialización (¿Qué valores toman al empezar?):** Por defecto en PyTorch (`nn.Embedding`), las coordenadas de los campeones se inicializan de forma aleatoria con valores procedentes de una **distribución normal estándar** (con media $\mu = 0$ y desviación estándar $\sigma = 1$). Esto significa que al principio son números decimales pequeños (tanto positivos como negativos, ej: $0.15, -0.42, 1.12$, etc.). El índice 0 (que reservamos en el vocabulario como "desconocido" o relleno) se inicializa con todos sus valores a **cero** y no se modifica durante el entrenamiento (`padding_idx=0`).
*   **El forward pass:** Cuando se procesa una partida, la red mira la tarjeta de los 10 campeones del draft, concatena sus coordenadas y hace una predicción de roaming.
*   **El backward pass (El aprendizaje):** Si el modelo se equivoca, calcula el error. La regla de la cadena (backpropagation) nos dice en qué dirección debemos mover los 16 números de las tarjetas de los campeones que han jugado para que el error disminuya la próxima vez.
*   **Convergencia semántica:** Si Thresh y Pyke son campeones que facilitan el roaming, el gradiente empujará sus tarjetas en la misma dirección. Al final del entrenamiento, si calculas la distancia entre sus vectores de 16 dimensiones, verás que están muy cerca en el espacio, mientras que el de Yuumi estará muy lejos.

---

### B. Mecánica de los Embeddings Compartidos vs. Por Rol
1.  **Shared Embeddings:** Hay una única matriz de tarjetas para todos. El vector de Thresh es el mismo si es tu support o si es el top enemigo. La red debe aprender a interpretarlo según la posición del input.
2.  **Per-Role Embeddings:** Hay 10 matrices independientes (una por posición del draft). Thresh-como-support-aliado y Thresh-como-support-enemigo tienen tarjetas independientes. Es mucho más expresivo.
3.  **Per-Role + Interactions:** Además de las tarjetas individuales, calculamos el **producto escalar (dot product)** entre las tarjetas del support aliado y el ADC aliado (sinergia) y contra el support enemigo (matchup). Esto obliga geométricamente a que los vectores se alineen si cooperan bien.

---

### C. Valor en el Proyecto
*   Al reducir la dimensión de 173 (One-Hot) a 16 (Embedding), creamos un **cuello de botella de información**. 
*   Esto actúa como un regularizador potente: la red ya no tiene parámetros suficientes para memorizar combinaciones raras, lo que mitiga el overfitting (el mejor modelo pasa de la época 6 a la 18).
*   En la memoria demostraremos que las distancias geométricas en el espacio de 16 dimensiones corresponden a roles de campeones reales en el juego sin haberle dado ninguna información previa al modelo.

---

## 5️⃣ Pérdida Robusta: Huber Loss vs. MSE

> [!NOTE]
> **Tu respuesta clave:** League of Legends tiene mucho ruido de partidas anómalas (desconexiones, jugadores rindiéndose, etc.). El **MSE penaliza los errores grandes al cuadrado**, haciendo que una sola partida rota arruine el aprendizaje. El **Huber Loss penaliza linealmente los errores grandes**, limitando su impacto.

### A. Explicación Conceptual
Imagina que en una partida limpia el support roamea un poco y el score esperado es 0.3. Pero resulta que su ADC se enfada y empieza a morir a propósito (ejercicio no cooperativo / *"troleo"*). El support se ve obligado a abandonar la botlane, dando un score observado de 0.9. El error es de 0.6.
*   **Bajo MSE (Mean Squared Error):** El error se eleva al cuadrado ($0.6^2 = 0.36$), y el gradiente resultante es muy grande. La red da un "bandazo" intentando corregir ese error enorme, desestructurando lo que había aprendido de las partidas normales.
*   **Bajo Huber Loss:** Si el error supera un límite (fijado en $\delta = 0.1$), el error ya no se eleva al cuadrado. Se trata con una penalización lineal (el gradiente es constante, $\pm 0.1$). Así, la partida rota se penaliza, pero su gradiente no es lo suficientemente grande como para desviar los pesos del modelo.

---

### B. Comparativa Matemática del Gradiente (El "empuje" del error)

| Tamaño del error ($e_i$) | Gradiente en MSE | Gradiente en Huber ($\delta = 0.1$) |
| :--- | :--- | :--- |
| **Pequeño** (ej. $e_i = 0.05$) | $0.05$ *(actualización suave)* | $0.05$ *(actualización suave)* |
| **Grande** (ej. $e_i = 0.80$) | **$0.80$** *(fuerza desmedida)* | **$0.10$** *(acotado / recortado)* |

---

### C. Valor en el Proyecto
*   Esto nos permite lidiar con la naturaleza ruidosa del juego. 
*   Además, lo combinamos con la ponderación de pesos de muestra (`sample_weight`): las partidas marcadas como caóticas (`chaos_flag = True`) reciben un peso de **0.40** (optimizado mediante un barrido experimental sistemático en validación), reduciendo su influencia en un 60% adicional en el gradiente final.
