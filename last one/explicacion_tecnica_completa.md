# Explicación Técnica Completa — Arquitecturas, ICC y Métricas

---

## Parte 1: Arquitecturas de los modelos

### 1.1. Visión general — ¿Qué reciben y qué producen todos los modelos?

Todos los modelos del proyecto reciben **la misma entrada** y producen **la misma salida**:

```
ENTRADA (pre-partida):
  • 10 IDs de campeón (5 aliados × {top, jungle, mid, bottom, utility}
                       + 5 enemigos × lo mismo)
  • 1 variable de lado del mapa (blue = 0, red = 1)

SALIDA:
  • 1 número continuo en [0, 1] → predicción del support_roam_score
```

Lo que cambia entre modelos es **cómo representan internamente los campeones** y **cómo combinan esa información** para producir la predicción.

```mermaid
graph LR
    A["🎮 Draft\n10 champion IDs\n+ side"] --> B{"¿Cómo representar\nlos campeones?"}
    B --> C["One-Hot\n(binario, 173 dims)"]
    B --> D["Embedding compartido\n(denso, 16 dims)"]
    B --> E["Embedding por rol\n(denso, 16 dims × 10)"]
    B --> F["Ordinal + Árboles\n(categórico nativo)"]
    C --> G["MLP OneHot"]
    D --> H["MLP Embed"]
    E --> I["MLP Per-Role"]
    F --> J["HistGBT"]
    G --> K["📊 Predicción\nroam_score ∈ ·0, 1·"]
    H --> K
    I --> K
    J --> K
```

---

### 1.2. Baseline: Champion Mean

El modelo más simple posible. No es un modelo de ML: es una **tabla de lookup**.

```mermaid
graph LR
    A["Input:\nally_utility = Thresh"] --> B["Tabla de medias\n(calculada en train)"]
    B --> C["Thresh → 0.42\nYuumi → 0.15\nBard → 0.58\nPyke → 0.51\n..."]
    C --> D["Predicción: 0.42"]
    
    style B fill:#f9f,stroke:#333,stroke-width:2px
```

**Cómo funciona:**
1. Durante "entrenamiento": calcula la media de `support_roam_score` para cada campeón de support en el dataset de train
2. Durante predicción: dado un draft, busca qué campeón juega de support aliado y devuelve su media histórica
3. Si el campeón no se vio en train → devuelve la media global

**¿Por qué es importante?** Porque marca un **suelo de comparación**. Si un modelo sofisticado no supera una tabla de medias, no está aprendiendo nada útil del resto del draft.

| Métrica (test) | Valor |
|---|---:|
| R² | 0.125 |
| Spearman | 0.336 |
| MAE | 0.144 |

---

### 1.3. HistGradientBoosting (GBT) — El modelo tabular

#### ¿Qué es un Gradient Boosted Tree?

Es un **conjunto (ensemble) de árboles de decisión** que se construyen de forma secuencial. Cada nuevo árbol intenta corregir los errores del conjunto anterior.

```mermaid
graph TD
    subgraph "Iteración 1"
        A1["Árbol 1"] --> P1["Predicción 1\n(muy aproximada)"]
    end
    P1 --> R1["Residuo 1\n= real - pred1"]
    
    subgraph "Iteración 2"
        R1 --> A2["Árbol 2\n(aprende del error)"]
        A2 --> P2["Predicción 2\n= pred1 + lr × árbol2"]
    end
    P2 --> R2["Residuo 2\n= real - pred2"]
    
    subgraph "Iteración N"
        R2 --> AN["Árbol N\n(corrige errores\nresidules)"]
        AN --> PN["Predicción final\n= Σ árboles × lr"]
    end
    
    style PN fill:#2d6,stroke:#333,stroke-width:2px
```

#### ¿Cómo trata los campeones?

`HistGradientBoosting` de sklearn tiene **soporte nativo para categóricas**. No necesita one-hot ni embeddings. Internamente:

1. Ordena los campeones por su valor medio del target
2. En cada split del árbol, agrupa campeones en dos subconjuntos
3. Busca el split que maximice la reducción de error

```
Ejemplo de un nodo del árbol:

         ¿ally_utility ∈ {Bard, Pyke, Thresh, Alistar}?
              /                          \
            SÍ                           NO
        (roamers)                    (enchanters/otros)
        score ↑ +0.08                score ↓ -0.03
```

#### Un árbol individual podría verse así:

```mermaid
graph TD
    ROOT["ally_utility ∈ {Bard,Pyke,Thresh,Alistar}?"]
    ROOT -->|Sí| L1["ally_bottom ∈ {Ezreal,Caitlyn}?"]
    ROOT -->|No| R1["enemy_utility ∈ {Leona,Nautilus}?"]
    
    L1 -->|Sí, ADC seguro| LL["🟢 +0.12\nADC con escape\n→ support puede irse"]
    L1 -->|No| LR["🟡 +0.06\nADC vulnerable\n→ algo de roam"]
    
    R1 -->|Sí, engage enemy| RL["🔴 -0.05\nContra engage\n→ difícil dejar bot"]
    R1 -->|No| RR["🟠 -0.01\nMatchup neutro"]
    
    style LL fill:#2d6
    style LR fill:#fd0
    style RL fill:#f44
    style RR fill:#fa0
```

> [!NOTE]
> En realidad el modelo usa **300 árboles** (max_iter=300) con profundidad máxima 6, cada uno con hasta 31 hojas. La predicción final es la suma ponderada de todos ellos. Cada árbol individual aporta poco, pero el conjunto captura interacciones complejas.

#### Hiperparámetros del GBT en el proyecto

| Parámetro | Valor | Significado |
|---|---:|---|
| max_iter | 300 | Número de árboles |
| max_depth | 6 | Profundidad máxima por árbol |
| learning_rate | 0.05 | Cuánto "peso" tiene cada nuevo árbol |
| min_samples_leaf | 50 | Mínimo de observaciones por hoja |
| max_leaf_nodes | 31 | Máximo de hojas por árbol |

#### Ventajas del GBT para este problema
- Maneja categóricas de forma nativa (no necesita one-hot ni embeddings)
- Captura interacciones no lineales automáticamente (p.ej. "Thresh + Ezreal → mucho roam" sin necesidad de codificarlo explícitamente)
- No necesita normalización, es robusto a outliers
- Rápido de entrenar (~30 segundos en 268K filas)

---

### 1.4. MLP OneHot — La red neuronal básica

#### ¿Qué es un MLP?

Un **Multi-Layer Perceptron** es una red neuronal feedforward compuesta de capas de neuronas densamente conectadas.

#### ¿Cómo codifica los campeones?

Cada campeón se convierte en un **vector binario one-hot** de dimensión 173 (= número de campeones en el dataset).

```
Ejemplo: Thresh tiene ID → índice 87 en el vocabulario

One-hot(Thresh) = [0, 0, 0, ..., 0, 1, 0, ..., 0]
                   ↑                 ↑ posición 87
                   173 dimensiones
```

Para los 10 slots del draft, se concatenan 10 vectores one-hot + el lado:

```
Input = [one_hot(ally_top) | one_hot(ally_jg) | ... | one_hot(enemy_util) | side]
         ←── 173 ──→         ←── 173 ──→               ←── 173 ──→        ← 1 →
                                                                    
Total: 10 × 173 + 1 = 1.731 dimensiones de entrada
```

#### Arquitectura completa

```mermaid
graph TD
    subgraph "Entrada: 10 champion slots + side"
        I1["ally_top\none-hot (173)"]
        I2["ally_jungle\none-hot (173)"]
        I3["..."]
        I4["enemy_utility\none-hot (173)"]
        I5["side (1)"]
    end
    
    I1 --> CONCAT["Concatenar\n→ vector de 1.731 dims"]
    I2 --> CONCAT
    I3 --> CONCAT
    I4 --> CONCAT
    I5 --> CONCAT
    
    CONCAT --> L1["Capa Lineal 1\n1.731 → 192 neuronas"]
    L1 --> R1["ReLU"]
    R1 --> BN1["BatchNorm"]
    BN1 --> D1["Dropout (0.35)"]
    
    D1 --> L2["Capa Lineal 2\n192 → 96 neuronas"]
    L2 --> R2["ReLU"]
    R2 --> BN2["BatchNorm"]
    BN2 --> D2["Dropout (0.35)"]
    
    D2 --> L3["Capa de salida\n96 → 1"]
    L3 --> OUT["📊 Predicción\nroam_score"]
    
    style OUT fill:#2d6,stroke:#333,stroke-width:2px
```

#### Los componentes uno a uno

| Componente | ¿Qué hace? | ¿Por qué? |
|---|---|---|
| **Capa Lineal** | `y = Wx + b` — transformación lineal | Aprende combinaciones de features |
| **ReLU** | `max(0, x)` — activación no lineal | Permite al modelo aprender relaciones no lineales. Sin ella, N capas lineales equivalen a 1 sola |
| **BatchNorm** | Normaliza las activaciones por mini-batch | Estabiliza el entrenamiento, permite learning rates más altos |
| **Dropout (0.35)** | Apaga aleatoriamente el 35% de neuronas en cada paso | Regularización: evita que el modelo memorice el train. Fuerza redundancia |

#### Problema del One-Hot

Con 173 dimensiones por slot y 10 slots, la primera capa tiene **1.731 × 192 = 332.352 pesos** solo para esa conexión. Es un modelo enorme (351K parámetros totales) para la señal disponible, que es débil. Consecuencia: **overfitting rápido** (best epoch = 6 de 150).

Además, todos los campeones están **equidistantes** en el espacio one-hot:

```
distancia(Thresh, Pyke)  = √2   ← roamers similares
distancia(Thresh, Yuumi) = √2   ← totalmente opuestos
distancia(Pyke, Yuumi)   = √2   ← MISMA distancia

→ El modelo no tiene ninguna pista de qué campeones son similares.
  Debe aprender TODA la estructura desde cero.
```

---

### 1.5. MLP con Embeddings Compartidos — Aprendiendo representaciones

#### La idea clave

En lugar de usar vectores binarios de 173 dimensiones, **comprimimos** cada campeón en un vector denso de solo **16 dimensiones** que se aprende durante el entrenamiento.

#### ¿Qué es un Embedding técnicamente?

```python
self.embed = nn.Embedding(num_embeddings=173, embedding_dim=16, padding_idx=0)
```

Es una **tabla de lookup entrenable**: una matriz **E ∈ ℝ^{173 × 16}**.

```
     dim 1  dim 2  dim 3  ...  dim 16
      ↓      ↓      ↓           ↓
Thresh  [ 0.23, -0.41,  0.87, ...,  0.12]   ← vector de Thresh
Pyke    [ 0.19, -0.38,  0.91, ...,  0.08]   ← vector de Pyke (¡cercano a Thresh!)
Yuumi   [-0.72,  0.55, -0.33, ..., -0.61]   ← vector de Yuumi (¡lejos de ambos!)
Bard    [ 0.31, -0.29,  0.78, ...,  0.22]   ← vector de Bard (cercano a roamers)
...
173 filas × 16 columnas = 2.768 parámetros
```

#### ¿Cómo funciona la tabla de lookup?

```mermaid
graph LR
    ID["Input: champion_id = 87\n(Thresh)"] --> LOOKUP["Tabla E\n173 × 16"]
    LOOKUP --> VEC["Output: E·87· = ·0.23, -0.41, 0.87, ...·\n→ vector denso de 16 dims"]
    
    style LOOKUP fill:#f9f,stroke:#333,stroke-width:2px
```

Es equivalente matemáticamente a multiplicar one-hot × E:

```
one_hot(87) × E = [0,0,...,1,...,0] × E = fila 87 de E
```

Pero **la dimensión reducida (173 → 16) es la diferencia crucial**: fuerza al modelo a codificar información útil en solo 16 números. Esto actúa como un **cuello de botella informacional** que obliga a agrupar campeones similares.

#### ¿Cómo se entrenan los embeddings?

```mermaid
graph TD
    subgraph "Forward Pass"
        INPUT["champion_id = 87"] --> EMB["Embedding Lookup\nE·87· → v ∈ ℝ¹⁶"]
        EMB --> MLP["Red MLP\n(capas lineales)"]
        MLP --> PRED["Predicción: 0.35"]
    end
    
    REAL["Target real: 0.52"] --> LOSS["Pérdida MSE\n= (0.35 - 0.52)²\n= 0.0289"]
    PRED --> LOSS
    
    subgraph "Backward Pass (Backpropagation)"
        LOSS --> GRAD_MLP["∂L/∂W_mlp\n→ actualizar pesos MLP"]
        LOSS --> GRAD_EMB["∂L/∂E·87·\n→ actualizar vector de Thresh"]
    end
    
    GRAD_EMB --> UPDATE["E·87· ← E·87· - lr × ∂L/∂E·87·\n\nEl vector de Thresh se mueve\nen la dirección que reduce el error"]
    
    style LOSS fill:#f44,stroke:#333
    style UPDATE fill:#2d6,stroke:#333
```

> [!IMPORTANT]
> **El embedding se entrena con el mismo optimizador (AdamW) y la misma learning rate que el resto de la red.** No hay nada especial: forma parte de `model.parameters()` y recibe gradientes como cualquier otro peso.

#### ¿Por qué convergen los campeones similares?

**Intuición con ejemplo numérico:**

1. El modelo ve muchas partidas con Thresh (roamer) → aprende que su vector debe empujar la predicción hacia scores altos
2. También ve muchas partidas con Pyke (roamer) → aprende algo parecido
3. Como ambos empujan en la misma dirección, sus vectores convergen hacia la misma zona del espacio de 16 dimensiones
4. Yuumi (anti-roamer) empuja en dirección opuesta → su vector queda lejos

```
Después del entrenamiento:

                    dim 1
                      ↑
         Yuumi •      |
               Lulu • |
                      |
         ────────────────────→ dim 2
                      |
                      |    • Thresh
                      |  • Pyke    • Bard
                      |      • Alistar
```

#### Arquitectura completa del MLP Embed

```mermaid
graph TD
    subgraph "Entrada: 10 IDs de campeón + side"
        C1["ally_top (ID)"]
        C2["ally_jungle (ID)"]
        C3["..."]
        C4["enemy_utility (ID)"]
        S["side (0 o 1)"]
    end
    
    subgraph "Embedding COMPARTIDO (misma tabla E para todos)"
        C1 --> E1["E·ally_top· → 16 dims"]
        C2 --> E2["E·ally_jg· → 16 dims"]
        C3 --> E3["..."]
        C4 --> E4["E·enemy_util· → 16 dims"]
    end
    
    E1 --> CONCAT["Concatenar\n10 × 16 + 1 = 161 dims"]
    E2 --> CONCAT
    E3 --> CONCAT
    E4 --> CONCAT
    S --> CONCAT
    
    CONCAT --> L1["Capa Lineal: 161 → 192"]
    L1 --> R1["ReLU + BatchNorm + Dropout"]
    R1 --> L2["Capa Lineal: 192 → 96"]
    L2 --> R2["ReLU + BatchNorm + Dropout"]
    R2 --> OUT["Capa salida: 96 → 1\n📊 roam_score"]
    
    style OUT fill:#2d6,stroke:#333,stroke-width:2px
```

**Parámetros totales: ~53K** (vs 351K del OneHot). 6.6× menos parámetros.

#### Limitación: "Compartido" = mismo vector para todos los roles

El vector de Thresh es el **mismo** independientemente de si Thresh es:
- Ally support (su rol natural → muy relevante)
- Enemy support (matchup → relevante de otra manera)
- Ally top (off-meta → casi irrelevante)

La red debe aprender a interpretar el mismo vector en contextos diferentes solo por su **posición** en el input concatenado.

---

### 1.6. MLP Per-Role + Interactions — El modelo más expresivo

#### La mejora: un embedding diferente para cada slot

En lugar de 1 tabla compartida, se crean **10 tablas independientes** (una por slot del draft):

```mermaid
graph TD
    subgraph "10 tablas de embedding independientes"
        T1["E_ally_top\n173 × 16"]
        T2["E_ally_jungle\n173 × 16"]
        T3["E_ally_mid\n173 × 16"]
        T4["E_ally_bottom\n173 × 16"]
        T5["E_ally_utility\n173 × 16"]
        T6["E_enemy_top\n173 × 16"]
        T7["E_enemy_jungle\n173 × 16"]
        T8["E_enemy_mid\n173 × 16"]
        T9["E_enemy_bottom\n173 × 16"]
        T10["E_enemy_utility\n173 × 16"]
    end
```

Ahora, **Thresh como ally support** y **Thresh como enemy support** tienen vectores completamente diferentes:

```
E_ally_utility[Thresh]  = [0.23, -0.41, 0.87, ...]   ← "Thresh en mi equipo como support"
E_enemy_utility[Thresh] = [0.11,  0.33, 0.22, ...]   ← "Thresh en el equipo rival como support"

→ El modelo puede aprender que "tener Thresh de support aliado"
  y "enfrentarse a Thresh enemigo" tienen efectos diferentes
  sobre el roaming de MI support.
```

#### Las interacciones explícitas (Dot Products)

Además de los embeddings por rol, se calculan **2 productos punto** que capturan relaciones entre slots:

```mermaid
graph TD
    subgraph "Embeddings por rol (10 × 16 = 160 dims)"
        E_AS["E_ally_utility·sup·\n= v_sup ∈ ℝ¹⁶"]
        E_AB["E_ally_bottom·adc·\n= v_adc ∈ ℝ¹⁶"]
        E_ES["E_enemy_utility·esup·\n= v_esup ∈ ℝ¹⁶"]
    end
    
    E_AS --> DOT1["Dot Product 1\nv_sup · v_esup\n= Σᵢ v_sup·i· × v_esup·i·\n→ 1 número"]
    E_ES --> DOT1
    
    E_AS --> DOT2["Dot Product 2\nv_sup · v_adc\n= Σᵢ v_sup·i· × v_adc·i·\n→ 1 número"]
    E_AB --> DOT2
    
    DOT1 --> MEANING1["Matchup support\n¿Compatibles o contrarios?"]
    DOT2 --> MEANING2["Sinergia botlane\n¿El ADC permite roamear?"]
    
    style DOT1 fill:#f9f
    style DOT2 fill:#f9f
```

**¿Qué mide el dot product?**

```
dot(v_sup, v_esup) = alto positivo → los vectores apuntan en la misma dirección
                                    → "matchup neutral / similar"
                   = alto negativo → vectores opuestos
                                    → "matchup polarizado"
                   = cercano a 0   → vectores ortogonales
                                    → "sin relación clara"
```

Es una medida de **similitud bilineal** aprendida entre los dos campeones, codificada en sus embeddings.

#### Arquitectura completa

```mermaid
graph TD
    subgraph "Entrada"
        C["10 champion IDs"]
        S["side (1)"]
    end
    
    C --> EMB["10 Embedding lookups\n(tablas independientes)"]
    
    EMB --> FLAT["Flatten\n10 × 16 = 160 dims"]
    EMB --> D1["dot·ally_sup, enemy_sup·\n→ matchup (1 dim)"]
    EMB --> D2["dot·ally_sup, ally_adc·\n→ sinergia (1 dim)"]
    
    FLAT --> CAT["Concatenar\n160 + 1 + 1 + 1 = 163 dims"]
    S --> CAT
    D1 --> CAT
    D2 --> CAT
    
    CAT --> L1["Linear: 163 → 192 + ReLU + BN + Dropout"]
    L1 --> L2["Linear: 192 → 96 + ReLU + BN + Dropout"]
    L2 --> OUT["Linear: 96 → 1\n📊 roam_score"]
    
    style OUT fill:#2d6,stroke:#333,stroke-width:2px
```

**Parámetros totales: ~78K** (53K embed + 25K cabeza MLP).

---

### 1.7. Comparativa visual de las 4 representaciones

````carousel
### One-Hot: todos equidistantes

```
Thresh  [0,0,...,1,...,0]  ─── √2 ───  Pyke  [0,0,...,1,...,0]
                            \                /
                          √2  \            / √2
                               \          /
                            Yuumi [0,0,...,1,...,0]

→ El modelo NO SABE que Thresh y Pyke son similares.
  Debe aprenderlo desde cero con 332K parámetros.
```

<!-- slide -->

### Embedding Compartido: similitudes aprendidas

```
         espacio de 16 dimensiones
         
         Yuumi ●          ← lejos de roamers
         Lulu ●
         
         ─────────────────────────
         
                    ● Thresh    ← cerca entre sí
                  ● Pyke
                ● Bard
              ● Alistar
         
→ MISMA representación sin importar el rol/lado.
  2.768 parámetros para la tabla.
```

<!-- slide -->

### Embedding Per-Role: especialización por posición

```
Tabla ally_utility:        Tabla enemy_utility:
  Thresh → [0.23, -0.41...]  Thresh → [0.11, 0.33...]
  Pyke   → [0.19, -0.38...]  Pyke   → [0.05, 0.28...]
  
→ "Thresh como MI support" ≠ "Thresh como support RIVAL"
  10 × 2.768 = 27.680 parámetros para embeddings.
```

<!-- slide -->

### HistGBT: categóricas nativas, sin representación explícita

```
       ¿ally_utility ∈ {Bard, Pyke, Thresh}?
              /                    \
            SÍ                     NO
     ¿ally_bottom ∈ {Ez,Cait}?   ¿enemy_util ∈ {Leona,Naut}?
        /        \                    /          \
      +0.12    +0.06              -0.05        -0.01

→ El árbol descubre agrupaciones óptimas automáticamente.
  No necesita representación vectorial.
  300 árboles × max 31 hojas cada uno.
```
````

### 1.8. Tabla resumen de arquitecturas

| | Champion Mean | HistGBT | MLP OneHot | MLP Embed | MLP Per-Role |
|---|---|---|---|---|---|
| **Tipo** | Lookup table | Ensemble de árboles | Red neuronal | Red neuronal | Red neuronal |
| **Representación** | — | Categórica nativa | One-hot (173d) | Embedding compartido (16d) | Embedding por rol (16d × 10) |
| **Input dim** | 1 | 31 categóricas | 1.731 | 161 | 163 |
| **Parámetros** | 0 | ~300 árboles | ~351K | ~53K | ~78K |
| **Interacciones** | Ninguna | Automáticas (splits) | Implícitas (capas) | Implícitas (capas) | Explícitas (dot) + implícitas |
| **Best epoch** | — | — | 6/150 ⚠️ | 18/150 | 17/150 |
| **R² (test)** | 0.125 | 0.160 | 0.155 | 0.150 | 0.154 |
| **Spearman (test)** | 0.336 | 0.387 | 0.381 | 0.376 | 0.381 |

> [!WARNING]
> El best epoch de 6/150 en MLP OneHot indica **overfitting rápido**: el modelo tiene 351K parámetros para una señal débil (R²≈0.16). En 6 epochs aprende la señal y luego empieza a memorizar ruido. Los embeddings mitigan esto al tener 6-7× menos parámetros.

---

## Parte 2: El ICC (Coeficiente de Correlación Intraclase)

### 2.1. ¿Qué problema resuelve el ICC?

Pregunta central: **¿Cuánta variabilidad del roam_score se debe a la composición del draft y cuánta a la ejecución individual de cada partida?**

Si agrupamos todas las partidas que tienen **la misma botlane** (mismo support + mismo ADC), ¿los scores dentro de cada grupo son consistentes o varían mucho?

```mermaid
graph TD
    subgraph "Grupo: Thresh + Ezreal (104 partidas)"
        G1A["Partida 1: score = 0.52"]
        G1B["Partida 2: score = 0.31"]
        G1C["Partida 3: score = 0.68"]
        G1D["Partida 4: score = 0.44"]
        G1E["..."]
        G1F["Partida 104: score = 0.39"]
        G1M["Media grupo = 0.45"]
    end
    
    subgraph "Grupo: Yuumi + Jinx (87 partidas)"
        G2A["Partida 1: score = 0.08"]
        G2B["Partida 2: score = 0.22"]
        G2C["Partida 3: score = 0.12"]
        G2D["Partida 4: score = 0.91 ⚠️ caótica"]
        G2E["..."]
        G2F["Partida 87: score = 0.15"]
        G2M["Media grupo = 0.18"]
    end
    
    G1M --> Q["¿Las medias de grupo son MUY diferentes entre sí?\n¿O los valores DENTRO de cada grupo varían demasiado?"]
    G2M --> Q
    Q --> ICC["ICC responde esta pregunta\ncon un solo número"]
    
    style ICC fill:#f9f,stroke:#333,stroke-width:2px
```

### 2.2. La descomposición de varianza (ANOVA)

El ICC se basa en descomponer la varianza total del roam_score en dos partes:

```mermaid
pie title "Varianza total del roam_score"
    "Varianza ENTRE grupos (13.9%)" : 13.9
    "Varianza DENTRO de grupos (86.1%)" : 86.1
```

```
Varianza TOTAL = Varianza ENTRE grupos + Varianza DENTRO de grupos
                 (diferencias entre          (diferencias entre partidas
                  composiciones)              con la MISMA composición)
```

### 2.3. Fórmula del ICC(1)

Se calcula mediante ANOVA de un factor:

```
ICC(1) = (MSB - MSW) / (MSB + (k̄ - 1) × MSW)
```

Donde:
- **MSB** = Mean Square Between groups = varianza entre las medias de los grupos
- **MSW** = Mean Square Within groups = varianza dentro de cada grupo (promediada)
- **k̄** = tamaño medio de grupo

#### Ejemplo numérico con datos del proyecto

```
Agrupación: botlane_champions (support + ADC)
  → ~3.846 grupos únicos
  → ~67 partidas por grupo de media

  MSB (varianza entre Thresh+Ez, Yuumi+Jinx, ...) = X
  MSW (varianza dentro de las 67 partidas de Thresh+Ez) = Y

  ICC = (X - Y) / (X + 66 × Y) = 0.139
```

**ICC = 0.139 significa**: solo el **13.9%** de la variabilidad total del roam_score se explica por qué botlane se jugó. El **86.1%** restante varía entre partidas con la misma composición.

### 2.4. ICC vs R² Group-Mean: ¿Cuál es la diferencia?

Son dos formas de responder a la misma pregunta, pero con matices:

```mermaid
graph TD
    subgraph "ICC (estadístico)"
        ICC_I["Descomposición ANOVA\nde la varianza"]
        ICC_I --> ICC_R["ICC = 0.139\n= proporción de varianza\nentre-grupos"]
    end
    
    subgraph "R² Group-Mean (predictivo)"
        R2_I["Para cada observación,\npredecir la media de su grupo"]
        R2_I --> R2_R["R² = 1 - SS_res/SS_tot\n= 0.173"]
    end
    
    ICC_R --> DIFF["¿Por qué son diferentes?"]
    R2_R --> DIFF
    
    DIFF --> EXPL["El R² group-mean está inflado porque\nse calcula IN-SAMPLE: las medias de grupo\nse calculan y evalúan sobre los MISMOS datos.\nGrupos pequeños sobreajustan su media."]
    
    style EXPL fill:#f44,stroke:#333,stroke-width:1px,color:#fff
```

| | ICC | R² Group-Mean |
|---|---|---|
| **Qué mide** | Consistencia intraclase (varianza explicada teórica) | Varianza explicada al predecir la media del grupo |
| **Sesgo** | Corregido por ANOVA (ajusta por tamaño de grupo) | **Sesgado hacia arriba** (in-sample) |
| **Valor (botlane+side)** | 0.139 | 0.173 |
| **Interpretación** | Más conservador, más honesto | Optimista, especialmente con grupos pequeños |
| **Cuál usar como "techo"** | ✅ Más apropiado | ⚠️ Solo si se recalcula out-of-sample |

> [!IMPORTANT]
> **Para el tutor**: "El ICC y el R² group-mean son dos medidas relacionadas pero distintas. El ICC (0.139) estima la proporción de varianza que es estable por composición, corregida por ANOVA. El R² (0.173) mide cuánto predice una tabla de medias por grupo, pero tiene sesgo in-sample. No 'sacamos un R² del ICC' — son cálculos paralelos sobre las mismas agrupaciones."

### 2.5. ¿Qué significan los resultados del ICC para el proyecto?

```mermaid
graph LR
    subgraph "Varianza del roam_score = 100%"
        A["13.9% explicable\npor composición\n(ICC)"]
        B["86.1% depende\nde la ejecución\n(irreductible desde draft)"]
    end
    
    A --> MODEL["Tu mejor modelo\ncaptura R²=0.161\n→ 11.6pp de 13.9pp"]
    B --> NOISE["Ningún modelo puede\npredecir esta parte\ncon info pre-partida"]
    
    MODEL --> GAP["Gap restante:\n13.9 - 16.1 ≈ -2pp (?)\nEl modelo captura MÁS\nque el ICC sugiere porque\nusa 10 slots, no solo botlane"]
    
    style B fill:#f44,stroke:#333,color:#fff
    style MODEL fill:#2d6,stroke:#333
```

> [!NOTE]
> Que el modelo (R²=0.161) supere el ICC de botlane (0.139) no es una contradicción. El ICC de botlane solo considera 2 campeones. El modelo usa **10 campeones + lado** → puede capturar señal adicional de los otros 8 slots. Lo coherente es que el R² del modelo quede **entre** el ICC de botlane (0.139) y el R² group-mean de botlane+side (0.173).

---

## Parte 3: ¿Por qué R² y Spearman? (y MAE)

### 3.1. Las tres métricas miden cosas diferentes

```mermaid
graph TD
    subgraph "R² — ¿Cuánta varianza explico?"
        R2["R² = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²"]
        R2 --> R2I["Compara el error del modelo\ncon el error de predecir\nsiempre la media"]
        R2I --> R2V["R² = 0.161\n→ El modelo explica el 16.1%\nde la varianza del score"]
    end
    
    subgraph "Spearman — ¿Ordeno bien los drafts?"
        SP["ρ = correlación de Pearson\nsobre los RANGOS"]
        SP --> SPI["No importa el valor exacto.\n¿Los drafts con scores altos\nreciben predicciones altas?"]
        SPI --> SPV["Spearman = 0.388\n→ Correlación moderada\nentre ranking real y predicho"]
    end
    
    subgraph "MAE — ¿Cuánto me equivoco en promedio?"
        MAE["MAE = media de |yᵢ - ŷᵢ|"]
        MAE --> MAEI["Error absoluto medio.\nMuy interpretable:\nen la escala del target."]
        MAEI --> MAEV["MAE = 0.141\n→ Me equivoco ±14.1\npuntos en escala 0-100"]
    end
    
    style R2V fill:#69f
    style SPV fill:#f96
    style MAEV fill:#6d6
```

### 3.2. ¿Por qué necesitamos las tres?

Cada una responde a una pregunta distinta y puede dar información diferente:

#### Escenario 1: R² alto pero Spearman bajo
```
Modelo A predice:  [0.40, 0.41, 0.39, 0.42, 0.38]
Valores reales:    [0.10, 0.90, 0.30, 0.70, 0.50]

→ R² puede ser positivo (las predicciones están "cerca de la media real")
→ Spearman ≈ 0 (el ORDEN está completamente mal)
→ El modelo colapsa a predecir siempre ~0.40 sin discriminar
```

#### Escenario 2: Spearman alto pero R² bajo
```
Modelo B predice:  [0.20, 0.80, 0.30, 0.70, 0.50]  (multiplicado × 2)
Valores reales:    [0.10, 0.40, 0.15, 0.35, 0.25]

→ Spearman = 1.0 (el ORDEN es perfecto)
→ R² < 0 (los valores están muy lejos de los reales)
→ El modelo ordena bien pero exagera las diferencias
```

#### En tu proyecto:

```
R² = 0.161   → "Explico el 16.1% de la varianza"
Spearman = 0.388 → "Ordeno razonablemente los drafts"
MAE = 0.141  → "En promedio me equivoco ±0.14 en [0,1]"

Las tres cuentan una historia COHERENTE:
→ Señal parcial, ranking moderado, error contenido.
```

### 3.3. ¿Por qué R² y no solo MSE/RMSE?

```
MSE = 0.0304    ← ¿Esto es bueno o malo? No tengo referencia.

R² = 1 - MSE/Var(y) = 1 - 0.0304/0.0363 = 0.161
                                    ↑
                              varianza del target

→ R² NORMALIZA el error por la varianza del fenómeno.
  0.161 significa "16.1% mejor que predecir siempre la media".
  Es INTERPRETABLE sin conocer la escala del target.
```

### 3.4. ¿Por qué Spearman y no Pearson?

```mermaid
graph TD
    subgraph "Pearson: relación LINEAL"
        P1["Asume que pred = a × real + b"]
        P1 --> P2["Sensible a outliers\ny a la escala"]
        P2 --> P3["Pearson = 0.402"]
    end
    
    subgraph "Spearman: relación MONÓTONA"
        S1["Solo mira si el ORDEN\nse preserva"]
        S1 --> S2["Robusto a outliers\ne independiente de la escala"]
        S2 --> S3["Spearman = 0.388"]
    end
    
    P3 --> COMP["Son similares aquí (0.402 vs 0.388)\n→ La relación es aproximadamente lineal"]
    S3 --> COMP
    
    COMP --> WHY["Spearman es MÁS relevante para este proyecto porque:\n1. La escala exacta del score es arbitraria (gamma 0.75, pesos)\n2. Lo que importa es ORDENAR drafts de menor a mayor roaming\n3. Es robusto a las partidas caóticas con scores extremos"]
    
    style WHY fill:#2d6,stroke:#333
```

### 3.5. ¿Y el MAE? ¿Por qué darle más importancia?

El MAE es la métrica **más interpretable** para el tribunal:

```
MAE = 0.141 en escala [0, 1]

Traducción: "En promedio, la predicción del modelo se desvía
            0.14 puntos del valor observado. En una escala
            de 0 a 100, es un error de ±14 puntos."

Comparación:
  • Global Mean:    MAE = 0.155  (predecir siempre la media)
  • Champion Mean:  MAE = 0.144  (tabla de medias por campeón)
  • Mejor modelo:   MAE = 0.141  (HistGBT)

→ Mejora absoluta sobre Champion Mean: solo 0.003 (3 milésimas)
→ Mejora absoluta sobre Global Mean:   0.014 (1.4 puntos %)

El MAE es HONESTO: muestra que la mejora incremental del ML
sobre una tabla de medias es pequeña en términos absolutos.
```

> [!TIP]
> **Métrica complementaria más intuitiva**: "El 74.2% de las predicciones están dentro de ±0.20 del valor real, y el 41.8% dentro de ±0.10". Esto es mucho más fácil de entender para un tribunal que "R²=0.161".

### 3.6. Resumen: qué aporta cada métrica

| Métrica | Pregunta que responde | Valor | Interpretación |
|---|---|---:|---|
| **R²** | ¿Cuánta varianza explico vs. predecir la media? | 0.161 | 16.1% de varianza explicada — cerca del techo (~0.17) |
| **Spearman** | ¿Ordeno bien los drafts de menos a más roaming? | 0.388 | Correlación moderada — el ranking tiene sentido |
| **MAE** | ¿Cuánto me equivoco en promedio? | 0.141 | Error de ±14.1 puntos en [0, 100] |
| **within ±0.10** | ¿Cuántas predicciones son "muy precisas"? | 41.8% | 4 de cada 10 predicciones aciertan ±10 puntos |
| **within ±0.20** | ¿Cuántas predicciones son "razonables"? | 74.2% | 3 de cada 4 predicciones están dentro de ±20 puntos |

---

## Resumen visual final

```mermaid
graph TD
    subgraph "EL PROYECTO EN UN DIAGRAMA"
        DRAFT["🎮 Draft\n10 campeones + lado"] --> MODEL["Modelos\n(GBT / MLP / Embed)"]
        TIMELINE["📊 Timeline\nposiciones minuto a minuto"] --> LABEL["Etiqueta\nroam_score ∈ ·0,1·"]
        
        MODEL --> PRED["Predicción"]
        LABEL --> EVAL["Evaluación"]
        PRED --> EVAL
        
        EVAL --> R2["R² = 0.161\n16% varianza explicada"]
        EVAL --> SP["Spearman = 0.388\nOrden razonable"]
        EVAL --> MAE_M["MAE = 0.141\n±14 puntos de error"]
        
        ICC_BOX["ICC = 0.139\n→ Techo teórico: ~14%\nde varianza es estable\npor composición"]
        
        ICC_BOX --> CONCLUSION["CONCLUSIÓN:\nEl modelo captura\ncasi toda la señal\ndisponible en el draft.\nEl 86% restante es\nejecución individual."]
    end
    
    style CONCLUSION fill:#2d6,stroke:#333,stroke-width:2px
    style ICC_BOX fill:#f9f,stroke:#333
```
