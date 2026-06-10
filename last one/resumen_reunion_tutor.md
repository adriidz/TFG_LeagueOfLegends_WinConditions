# Resumen del Proyecto — Reunión con Tutor

## Pregunta de Investigación
¿Hasta qué punto la configuración de personajes elegidos **antes** de empezar la partida permite anticipar la movilidad temprana del agente de soporte?

---

## Qué hemos hecho

### 1. Pipeline de datos propio (ELT)
- Descarga concurrente desde la API de Riot Games (partidas + timelines).
- Filtrado: Solo/Duo ranked, EUW, rango Master+, parches 16.2–16.8.
- **383.104 observaciones** finales de tipo `(match_id, team_id)`.
- Procesamiento geométrico de coordenadas con polígonos manuales para clasificar la posición del soporte minuto a minuto.

### 2. Etiqueta construida (`support_roam_score`)
- Score continuo [0, 1] que combina: tiempo fuera de botlane (45%), distancia al tirador (35%), y gap de experiencia (20%). Transformación gamma 0.75.
- **Validación:** Correlación de Spearman de **0.82** con un ranking experto de 47 campeones de soporte.

### 3. Protocolo experimental riguroso
- Split train/val/test (70/15/15) **agrupado por match_id** (evitar fuga de datos).
- Pesos de muestra: partidas limpias = 1.0, partidas caóticas = **0.40** (peso optimizado sistemáticamente en validación).
- **3 seeds** (42, 123, 456) para verificar estabilidad.
- **Todos los modelos** evaluados con la misma entrada: 10 IDs de campeón + lado del mapa.

### 4. Prototipo CLI Aplicado
- Desarrollo de una herramienta interactiva de consola que asimila un *draft* manual y devuelve un percentil calibrado representativo de la tendencia esperada de la movilidad temprana de la *botlane*.

---

## Resultados clave

### Tabla principal (Test Set, OOS)

| Modelo | R² | Spearman | MAE |
|:---|:---:|:---:|:---:|
| Media global | -0.0008 | — | 15.51% |
| Media del campeón soporte | 0.1243 | 0.3362 | 14.38% |
| **HistGBT (mejor)** | **0.1595** | **0.3877** | **14.08%** |
| MLP OneHot | 0.1536 | 0.3801 | 14.12% |
| MLP Per-Role + Inter. | 0.1527 | 0.3783 | 14.14% |

### Interpretación rápida
- **Hay señal en el draft**, todos los modelos superan la media global.
- La identidad del soporte es la variable dominante (R² = 0.1243 solo con ella).
- El draft completo (HistGBT) sube a R² = 0.1595 → el **contexto del resto del draft aporta un +0.035 de R²**.
- **HistGBT > Redes Neuronales:** El árbol tabular (HistGBT) supera a las MLPs porque gestiona mejor la señal débil combinada con alta cardinalidad (173 campeones), mitigando el sobreajuste que sufren las redes al mapear los embeddings.
- Los MAEs están comprimidos porque la señal es débil y el modelo predice cerca de la media → Spearman (0.3877) es la métrica clave para evaluar la ordenación.

### Experimento residual (prueba de que el draft aporta más allá del soporte)
1. Se aísla el efecto base del soporte (media suavizada, R² = 0.1240).
2. Se entrena un HistGBT sobre el **residuo** (target - media_support), usando solo el resto del draft.
3. El modelo residual obtiene R² = 0.0386, Spearman = 0.1892 **sobre el residuo**.
4. Modelo aditivo combinado: R² = 0.1584, Spearman = 0.3854.
5. **Lift neto del contexto del draft: +0.0343 R², +0.0498 Spearman.**

### Referencias OOS (group-mean calculada en train, evaluada en test)
- Support champion: R² = 0.1249
- Botlane: R² = 0.1239
- Botlane + side: R² = 0.1132
- **HistGBT: R² = 0.1595** → supera todas las referencias de lookup.

---

## Dudas del tutor resueltas

### "Los entrenamientos no son rigurosos"
- **Ahora sí:** Todos los modelos comparten la misma entrada (10 campeones + side), el mismo split, las mismas seeds y las mismas métricas. No hay comparaciones injustas.
- **Optimización sistemática:** El peso asignado a las partidas caóticas (`sample_weight = 0.40`) no fue arbitrario; se realizó un barrido paramétrico (sweep) completo de 0.0 a 1.0 en validación para encontrar el valor que maximiza la generalización sin descartar datos.

### "Los resultados están muy juntos entre sí y cerca de la media por campeón"
- Es esperable: la mayor parte de la señal predecible viene del soporte. El draft completo añade un +3.5% de R² y +5 puntos porcentuales de Spearman sobre la baseline de champion mean. Parece poco, pero el experimento residual demuestra que es señal real y no ruido.

### "¿De dónde sale el R² del ICC?"
- El ICC (Coeficiente de Correlación Intraclase) y el R² OOS son cosas distintas:
  - **ICC:** Estadístico descriptivo calculado in-sample (en train) mediante ANOVA. Mide la consistencia interna de los grupos.
  - **R² group-mean OOS:** Predicción real. Se entrena la media del grupo en train y se evalúa en test. Este es el comparable con los modelos.
  - Los dos se reportan por separado en la Tabla 2 del informe.

### "¿Por qué el 16% no es un fracaso?"
- Diagnóstico con 4 hipótesis descartadas: ni la arquitectura (108 configs de MLP), ni la codificación (embeddings vs one-hot), ni la fórmula de la etiqueta (15 variaciones, corr. ≥ 0.99 entre ellas), ni el target alternativo (etiqueta por eventos: R² = 0.091) mejoran el resultado.
- **El 16% es el techo estructural del problema**: el draft define predisposición, no ejecución real de la partida. Esto se corrobora empíricamente observando que el modelo alcanza un estimulante R² = 0.1719 en "partidas limpias" pero decae al R² = 0.1220 en "partidas caóticas" (donde subyace un exceso de fallos impredecibles).

---

## Conclusión en una frase
> La configuración prepartida contiene una señal parcial pero real y verificable (R² ≈ 16%, Spearman ≈ 0.39) sobre la movilidad temprana del soporte. El modelo aprende del draft completo, no solo del soporte, y el límite de predictibilidad es intrínseco al problema, no un fallo de diseño.
