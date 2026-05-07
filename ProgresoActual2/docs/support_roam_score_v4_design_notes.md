# Support roam score v4 - notas de diseno

Estas notas recogen la decision metodologica posterior al analisis de variantes
`v3`. La calibracion `gamma=0.75` queda como resultado secundario: mejora el
rango util de `v2`, pero no cambia la semantica de la etiqueta. La siguiente
linea de trabajo sera una etiqueta `v4` construida desde menor nivel:
geometria, distancia, XP y eventos observables del timeline.

## Diagnostico v3

- `v2` ordena razonablemente bien los campeones, pero comprime la escala hacia
  la izquierda.
- `v3_calibrated_gamma075` mejora rango sin romper ranking porque es una
  transformacion monotona de `v2`.
- Esa mejora debe presentarse como calibracion de rango, no como nueva
  definicion del fenomeno.
- Para una mejora real hay que revisar que se considera bot, roam, distancia y
  participacion observable.

## Objetivo v4

Medir la intensidad de roaming observado del support entre los minutos tempranos
sin depender de conocimiento experto ni de entrenamiento de modelo.

La etiqueta debe seguir siendo observacional:

- donde esta el support;
- cuanto se separa del ADC;
- cuanto se aleja de la zona de botlane;
- que ocurre entre frames de timeline en lo que el support pudo participar;
- si la separacion se refleja tambien en XP.

## Componentes propuestos

Formula inicial orientativa:

```text
support_roam_score_v4 =
  0.35 * spatial_roam_signal
+ 0.20 * adc_separation_signal
+ 0.15 * distance_from_bot_signal
+ 0.15 * event_participation_signal
+ 0.10 * xp_gap_signal
+ 0.05 * persistence_confidence_signal
```

Los pesos son una hipotesis inicial, no una verdad cerrada.

### 1. Geometria y senal espacial

Pulir la geometria antes de tocar mas componentes:

- refinar coordenadas de botlane, rio, jungla, mid y zonas de objetivos;
- distinguir botlane real, rio bot, rio mid, dragon, grubs/herald, jungla propia
  y jungla enemiga;
- excluir base/recalls;
- normalizar por side cuando haga falta para que blue/red sean comparables;
- sustituir parte de la logica binaria por distancia continua a botlane o al
  corredor de bot.

La idea es que no pese igual:

```text
tribush bot != rio/mid != grubs != base
```

### 2. Separacion del ADC

Mantener distancia al ADC como senal principal, idealmente continua:

```text
adc_separation_signal = sigmoid((dist_to_adc - threshold) / scale)
```

Esto evita umbrales bruscos tipo `2499` no roam y `2501` roam.

### 3. Distancia a botlane

Anadir una senal continua de distancia a botlane. No basta con estar fuera de
bot: importa cuanto y hacia donde se desplaza el support.

Ejemplos:

- cerca de bot pero fuera del poligono: senal baja/media;
- rio/mid/jungla lejos de bot: senal alta;
- base: excluida o muy baja.

### 4. Eventos del timeline

Los eventos deben medir evidencia de participacion, no solo exito.

No contar unicamente asistencias favorables. Tambien deben considerarse eventos
desfavorables o neutros en los que el support pudo estar implicado entre dos
frames:

- kill/assist fuera de bot;
- muerte del support fuera de bot;
- pelea en mid/rio/jungla/objetivo;
- presencia cerca de dragon, grubs o herald;
- participacion en objetivo neutral;
- muerte del ADC en bot mientras el support esta lejos, como posible roam
  costoso;
- acciones de vision fuera de bot si estan disponibles.

Separar conceptualmente:

```text
event_presence_score = hubo evento relevante con implicacion del support
event_outcome_score  = resultado favorable/desfavorable
```

Para la etiqueta de roaming debe pesar mas la presencia que el resultado.

### 5. XP gap

Mantener la diferencia/ratio de XP, pero como evidencia secundaria.

Razon: si el support abandona bot y el ADC sigue farmeando solo, puede aparecer
una brecha relativa de XP. Es una senal indirecta de separacion y tiempo fuera
de botlane.

Cautelas:

- puede verse afectada por muertes;
- recalls y waves meten ruido;
- el support puede roam sin generar gran gap;
- acompanhar al jungla puede no dar XP.

Por eso `xp_gap_signal` no deberia dominar la etiqueta.

### 6. Persistencia/confianza

No sustituir la etiqueta por "episodios" obligatorios de varios frames seguidos.
La resolucion del timeline es baja y los roams pueden ocurrir entre snapshots.

Mejor:

```text
spatial_base = senal frame a frame
persistence_confidence = plus pequeno si hay continuidad
```

Un frame aislado muy lejos de bot puede contar. La continuidad debe aumentar la
confianza, no ser requisito.

## Ablacion necesaria

Como v4 tendra mas componentes, debe defenderse con ablacion por bloques:

```text
v4_spatial
v4_spatial + adc_distance
v4_spatial + adc_distance + bot_distance
v4_spatial + adc_distance + bot_distance + xp
v4_spatial + adc_distance + bot_distance + xp + events
v4_full + persistence/confidence
```

Metricas a comparar:

- distribucion global;
- Pyke/Bard/Rakan/Pantheon frente a Yuumi/Milio/Lulu/Soraka;
- correlacion con referencia experta;
- separacion high-roam vs anchored;
- side bias;
- casos auditables.

Regla metodologica:

> Un componente solo se queda si mejora interpretabilidad o salud de etiqueta
> sin romper estabilidad, y si tiene una justificacion observacional
> independiente.

## Validacion humana

No validar manualmente cientos de partidas. Generar una muestra pequena y
explicable:

- top 20 scores v4;
- bottom 20 scores v4;
- casos donde v2 bajo pero v4 alto;
- casos donde v2 alto pero v4 bajo;
- casos raros: Yuumi alta, Bard baja, Pyke baja.

Para cada caso conviene guardar:

```text
match_id
team_id
support champion
ADC champion
score v2
score v4
frames fuera de bot
distancia media/maxima al ADC
eventos fuera de bot
zona principal
lectura automatica
```

Esto servira para interpretar si la etiqueta captura roaming real o si hay
errores de geometria/eventos.

## Siguiente paso

Empezar por geometria:

1. Auditar la definicion actual de zonas en `shared_utils.py`.
2. Visualizar ejemplos de posiciones del support/ADC por zona.
3. Revisar limites de botlane, rio, jungla y objetivos.
4. Proponer `geometry_v4` antes de anadir eventos.
