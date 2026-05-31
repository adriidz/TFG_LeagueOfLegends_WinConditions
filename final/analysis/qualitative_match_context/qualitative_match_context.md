# Qualitative Match Context for Top Errors

This file adds real match evidence from `match.json` and `timeline.json`: early kills, deaths, assists, objective events and final botlane stats. It is the qualitative layer on top of the label reconstruction.

| error_rank | side | patch  | ally_support | ally_adc | enemy_support | enemy_adc | prediction | actual | outside_ratio_v5 | far_ratio_v5 | support_early_kills | support_early_deaths | support_early_assists | adc_early_deaths | bot_related_kill_events_0_12 | label_diagnostic                      | qualitative_reading                                                                                                                                                                                                                                                                                               |
| ---------- | ---- | ------ | ------------ | -------- | ------------- | --------- | ---------- | ------ | ---------------- | ------------ | ------------------- | -------------------- | --------------------- | ---------------- | ---------------------------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1          | red  | 16.800 | Yuumi        | Smolder  | Pyke          | Velkoz    | 0.209      | 1.000  | 1.000            | 1.000        | 1                   | 4                    | 3                     | 7                | 21                           | consistent_full_roam_label            | modelo infrapredice un roaming extremo; la etiqueta parece consistente: fuera de bot y lejos del ADC casi toda la ventana; support muere 4 vez/veces antes de 12; ADC muere 7 vez/veces antes de 12; support participa en kills tempranas                                                                         |
| 2          | blue | 16.200 | Yuumi        | Zeri     | Sona          | Lucian    | 0.174      | 0.930  | 0.833            | 1.000        | 0                   | 3                    | 0                     | 6                | 11                           | mostly_outside_bot_context            | modelo infrapredice un roaming extremo; la etiqueta viene sobre todo de presencia fuera de bot; support muere 3 vez/veces antes de 12; ADC muere 6 vez/veces antes de 12                                                                                                                                          |
| 3          | red  | 16.200 | Senna        | Caitlyn  | Blitzcrank    | Tristana  | 0.310      | 1.000  | 1.000            | 1.000        | 0                   | 6                    | 2                     | 1                | 14                           | consistent_full_roam_label            | modelo infrapredice un roaming extremo; la etiqueta parece consistente: fuera de bot y lejos del ADC casi toda la ventana; support muere 6 vez/veces antes de 12; ADC muere 1 vez/veces antes de 12; support participa en kills tempranas                                                                         |
| 4          | red  | 16.500 | Senna        | Ashe     | Lulu          | Twitch    | 0.263      | 0.943  | 0.833            | 1.000        | 2                   | 4                    | 4                     | 3                | 14                           | mostly_outside_bot_context            | modelo infrapredice un roaming extremo; la etiqueta viene sobre todo de presencia fuera de bot; support muere 4 vez/veces antes de 12; ADC muere 3 vez/veces antes de 12; support participa en kills tempranas                                                                                                    |
| 5          | red  | 16.200 | Senna        | Kaisa    | Rell          | Ezreal    | 0.304      | 0.971  | 1.000            | 1.000        | 0                   | 3                    | 7                     | 2                | 15                           | consistent_full_roam_label            | modelo infrapredice un roaming extremo; la etiqueta parece consistente: fuera de bot y lejos del ADC casi toda la ventana; support muere 3 vez/veces antes de 12; ADC muere 2 vez/veces antes de 12; support participa en kills tempranas                                                                         |
| 6          | blue | 16.200 | Karma        | Caitlyn  | Lulu          | Ashe      | 0.337      | 1.000  | 1.000            | 1.000        | 0                   | 3                    | 6                     | 3                | 13                           | consistent_full_roam_label            | modelo infrapredice un roaming extremo; la etiqueta parece consistente: fuera de bot y lejos del ADC casi toda la ventana; support muere 3 vez/veces antes de 12; ADC muere 3 vez/veces antes de 12; support participa en kills tempranas                                                                         |
| 7          | blue | 16.800 | Sona         | Ashe     | Blitzcrank    | Jinx      | 0.341      | 1.000  | 1.000            | 1.000        | 2                   | 4                    | 4                     | 4                | 16                           | consistent_full_roam_label            | modelo infrapredice un roaming extremo; la etiqueta parece consistente: fuera de bot y lejos del ADC casi toda la ventana; support muere 4 vez/veces antes de 12; ADC muere 4 vez/veces antes de 12; support participa en kills tempranas                                                                         |
| 8          | red  | 16.200 | Yuumi        | Aphelios | Velkoz        | Caitlyn   | 0.158      | 0.817  | 0.667            | 0.833        | 0                   | 2                    | 1                     | 5                | 12                           | mostly_far_from_adc                   | la etiqueta viene sobre todo de distancia al ADC; support muere 2 vez/veces antes de 12; ADC muere 5 vez/veces antes de 12; support participa en kills tempranas                                                                                                                                                  |
| 9          | blue | 16.300 | Velkoz       | Twitch   | Nami          | Mel       | 0.342      | 1.000  | 1.000            | 1.000        | 4                   | 3                    | 1                     | 0                | 11                           | consistent_full_roam_label            | modelo infrapredice un roaming extremo; la etiqueta parece consistente: fuera de bot y lejos del ADC casi toda la ventana; support muere 3 vez/veces antes de 12; support participa en kills tempranas                                                                                                            |
| 10         | red  | 16.200 | Lux          | Caitlyn  | Karma         | Jhin      | 0.343      | 1.000  | 1.000            | 1.000        | 0                   | 4                    | 1                     | 3                | 18                           | low_valid_support_frames              | modelo infrapredice un roaming extremo; la etiqueta parece consistente: fuera de bot y lejos del ADC casi toda la ventana; support muere 4 vez/veces antes de 12; ADC muere 3 vez/veces antes de 12; support participa en kills tempranas; requiere cautela por diagnostico de etiqueta: low_valid_support_frames |
| 11         | blue | 16.800 | Nami         | Yasuo    | Soraka        | Smolder   | 0.320      | 0.977  | 1.000            | 1.000        | 0                   | 2                    | 2                     | 5                | 12                           | consistent_full_roam_label            | modelo infrapredice un roaming extremo; la etiqueta parece consistente: fuera de bot y lejos del ADC casi toda la ventana; support muere 2 vez/veces antes de 12; ADC muere 5 vez/veces antes de 12; support participa en kills tempranas                                                                         |
| 12         | red  | 16.700 | Sona         | Yunara   | Camille       | Corki     | 0.347      | 0.998  | 1.000            | 1.000        | 1                   | 5                    | 3                     | 1                | 16                           | low_valid_coop_frames                 | modelo infrapredice un roaming extremo; la etiqueta parece consistente: fuera de bot y lejos del ADC casi toda la ventana; support muere 5 vez/veces antes de 12; ADC muere 1 vez/veces antes de 12; support participa en kills tempranas; requiere cautela por diagnostico de etiqueta: low_valid_coop_frames    |
| 13         | red  | 16.200 | Yuumi        | Tristana | Milio         | Lucian    | 0.131      | 0.777  | 0.571            | 0.750        | 1                   | 1                    | 3                     | 4                | 14                           | possible_adc_death_base_coop_artifact | la etiqueta viene sobre todo de distancia al ADC; support muere 1 vez/veces antes de 12; ADC muere 4 vez/veces antes de 12; support participa en kills tempranas; requiere cautela por diagnostico de etiqueta: possible_adc_death_base_coop_artifact                                                             |
| 14         | red  | 16.200 | Braum        | Vayne    | Karma         | Varus     | 0.345      | 0.992  | 1.000            | 1.000        | 1                   | 3                    | 3                     | 3                | 13                           | consistent_full_roam_label            | modelo infrapredice un roaming extremo; la etiqueta parece consistente: fuera de bot y lejos del ADC casi toda la ventana; support muere 3 vez/veces antes de 12; ADC muere 3 vez/veces antes de 12; support participa en kills tempranas                                                                         |
| 15         | red  | 16.800 | Taric        | Lucian   | Yuumi         | Zeri      | 0.269      | 0.913  | 0.833            | 1.000        | 1                   | 1                    | 8                     | 1                | 11                           | mostly_outside_bot_context            | modelo infrapredice un roaming extremo; la etiqueta viene sobre todo de presencia fuera de bot; support muere 1 vez/veces antes de 12; ADC muere 1 vez/veces antes de 12; support participa en kills tempranas                                                                                                    |
| 16         | blue | 16.300 | Senna        | Veigar   | Blitzcrank    | Xerath    | 0.358      | 1.000  | 1.000            | 1.000        | 3                   | 4                    | 2                     | 2                | 17                           | consistent_full_roam_label            | la etiqueta parece consistente: fuera de bot y lejos del ADC casi toda la ventana; support muere 4 vez/veces antes de 12; ADC muere 2 vez/veces antes de 12; support participa en kills tempranas                                                                                                                 |
| 17         | red  | 16.500 | Ivern        | Lucian   | Senna         | Jhin      | 0.362      | 1.000  | 1.000            | 1.000        | 0                   | 5                    | 2                     | 2                | 16                           | consistent_full_roam_label            | la etiqueta parece consistente: fuera de bot y lejos del ADC casi toda la ventana; support muere 5 vez/veces antes de 12; ADC muere 2 vez/veces antes de 12; support participa en kills tempranas                                                                                                                 |
| 18         | red  | 16.600 | Braum        | Varus    | Nautilus      | Aphelios  | 0.360      | 0.996  | 1.000            | 1.000        | 1                   | 2                    | 1                     | 3                | 8                            | consistent_full_roam_label            | la etiqueta parece consistente: fuera de bot y lejos del ADC casi toda la ventana; support muere 2 vez/veces antes de 12; ADC muere 3 vez/veces antes de 12; support participa en kills tempranas                                                                                                                 |
| 19         | blue | 16.600 | Yuumi        | Corki    | Sona          | Zed       | 0.219      | 0.855  | 0.800            | 1.000        | 0                   | 2                    | 3                     | 6                | 16                           | low_valid_coop_frames                 | la etiqueta viene sobre todo de presencia fuera de bot; support muere 2 vez/veces antes de 12; ADC muere 6 vez/veces antes de 12; support participa en kills tempranas; requiere cautela por diagnostico de etiqueta: low_valid_coop_frames                                                                       |
| 20         | red  | 16.800 | Zyra         | Ashe     | Thresh        | Jinx      | 0.289      | 0.922  | 1.000            | 1.000        | 2                   | 3                    | 6                     | 8                | 22                           | consistent_full_roam_label            | modelo infrapredice un roaming extremo; la etiqueta parece consistente: fuera de bot y lejos del ADC casi toda la ventana; support muere 3 vez/veces antes de 12; ADC muere 8 vez/veces antes de 12; support participa en kills tempranas                                                                         |

## Case notes

The notes below are generated from real `CHAMPION_KILL` events in the Riot timeline. They are not inferred from the draft model.

### Case 1: EUW1_7831489390 (Yuumi + Smolder)

Predicho=0.209, real=1.000, error=0.791. Draft: Sion/Talon/KogMaw/Smolder/Yuumi vs Jayce/Shaco/Azir/Velkoz/Pyke.
Lectura: modelo infrapredice un roaming extremo; la etiqueta parece consistente: fuera de bot y lejos del ADC casi toda la ventana; support muere 4 vez/veces antes de 12; ADC muere 7 vez/veces antes de 12; support participa en kills tempranas. KDA final botlane aliada: support 1/8/8, ADC 2/8/1.

Eventos tempranos relevantes:
- min 1.36: Velkoz(BOTTOM,T100,pid4) mata a Yuumi(UTILITY,T200,pid10); assists: Pyke(UTILITY,T100,pid5) (muere support aliado)
- min 1.39: Smolder(BOTTOM,T200,pid9) mata a Pyke(UTILITY,T100,pid5); assists: Yuumi(UTILITY,T200,pid10) (asiste support aliado; participa ADC aliado)
- min 1.48: Velkoz(BOTTOM,T100,pid4) mata a Smolder(BOTTOM,T200,pid9); assists: Pyke(UTILITY,T100,pid5) (muere ADC aliado)
- min 3.03: Shaco(JUNGLE,T100,pid2) mata a Smolder(BOTTOM,T200,pid9); assists: Velkoz(BOTTOM,T100,pid4); Pyke(UTILITY,T100,pid5) (muere ADC aliado)
- min 3.87: Velkoz(BOTTOM,T100,pid4) mata a Smolder(BOTTOM,T200,pid9); assists: Shaco(JUNGLE,T100,pid2); Pyke(UTILITY,T100,pid5) (muere ADC aliado)
- min 3.99: Velkoz(BOTTOM,T100,pid4) mata a Yuumi(UTILITY,T200,pid10); assists: Pyke(UTILITY,T100,pid5) (muere support aliado)
- min 4.66: Yuumi(UTILITY,T200,pid10) mata a Shaco(JUNGLE,T100,pid2); assists: Sion(TOP,T200,pid6); Talon(JUNGLE,T200,pid7) (participa support aliado)
- min 6.65: Velkoz(BOTTOM,T100,pid4) mata a Smolder(BOTTOM,T200,pid9); assists: - (muere ADC aliado)

### Case 2: EUW1_7706461344 (Yuumi + Zeri)

Predicho=0.174, real=0.930, error=0.756. Draft: KSante/Viego/Cassiopeia/Zeri/Yuumi vs Camille/Graves/TwistedFate/Lucian/Sona.
Lectura: modelo infrapredice un roaming extremo; la etiqueta viene sobre todo de presencia fuera de bot; support muere 3 vez/veces antes de 12; ADC muere 6 vez/veces antes de 12. KDA final botlane aliada: support 0/5/0, ADC 0/9/0.

Eventos tempranos relevantes:
- min 1.81: Sona(UTILITY,T200,pid10) mata a Zeri(BOTTOM,T100,pid4); assists: Lucian(BOTTOM,T200,pid9) (muere ADC aliado)
- min 3.49: Sona(UTILITY,T200,pid10) mata a Zeri(BOTTOM,T100,pid4); assists: Lucian(BOTTOM,T200,pid9) (muere ADC aliado)
- min 4.34: Lucian(BOTTOM,T200,pid9) mata a Zeri(BOTTOM,T100,pid4); assists: Sona(UTILITY,T200,pid10) (muere ADC aliado)
- min 5.70: Lucian(BOTTOM,T200,pid9) mata a Yuumi(UTILITY,T100,pid5); assists: Sona(UTILITY,T200,pid10) (muere support aliado)
- min 6.19: Lucian(BOTTOM,T200,pid9) mata a Zeri(BOTTOM,T100,pid4); assists: Sona(UTILITY,T200,pid10) (muere ADC aliado)
- min 7.47: Graves(JUNGLE,T200,pid7) mata a Yuumi(UTILITY,T100,pid5); assists: - (muere support aliado)
- min 7.91: Graves(JUNGLE,T200,pid7) mata a Zeri(BOTTOM,T100,pid4); assists: Lucian(BOTTOM,T200,pid9); Sona(UTILITY,T200,pid10) (muere ADC aliado)
- min 9.27: Graves(JUNGLE,T200,pid7) mata a Zeri(BOTTOM,T100,pid4); assists: TwistedFate(MIDDLE,T200,pid8); Sona(UTILITY,T200,pid10) (muere ADC aliado)

### Case 3: EUW1_7708715292 (Senna + Caitlyn)

Predicho=0.310, real=1.000, error=0.690. Draft: Teemo/Diana/Yasuo/Caitlyn/Senna vs Swain/Nidalee/Qiyana/Tristana/Blitzcrank.
Lectura: modelo infrapredice un roaming extremo; la etiqueta parece consistente: fuera de bot y lejos del ADC casi toda la ventana; support muere 6 vez/veces antes de 12; ADC muere 1 vez/veces antes de 12; support participa en kills tempranas. KDA final botlane aliada: support 2/12/18, ADC 11/4/8.

Eventos tempranos relevantes:
- min 2.63: Tristana(BOTTOM,T100,pid4) mata a Senna(UTILITY,T200,pid10); assists: Blitzcrank(UTILITY,T100,pid5) (muere support aliado)
- min 2.71: Caitlyn(BOTTOM,T200,pid9) mata a Tristana(BOTTOM,T100,pid4); assists: Senna(UTILITY,T200,pid10) (asiste support aliado; participa ADC aliado)
- min 4.21: Tristana(BOTTOM,T100,pid4) mata a Caitlyn(BOTTOM,T200,pid9); assists: Blitzcrank(UTILITY,T100,pid5) (muere ADC aliado)
- min 4.51: Tristana(BOTTOM,T100,pid4) mata a Senna(UTILITY,T200,pid10); assists: Blitzcrank(UTILITY,T100,pid5) (muere support aliado)
- min 5.94: Diana(JUNGLE,T200,pid7) mata a Blitzcrank(UTILITY,T100,pid5); assists: Caitlyn(BOTTOM,T200,pid9) (asiste ADC aliado)
- min 5.96: Qiyana(MIDDLE,T100,pid3) mata a Senna(UTILITY,T200,pid10); assists: - (muere support aliado)
- min 6.18: Diana(JUNGLE,T200,pid7) mata a Tristana(BOTTOM,T100,pid4); assists: Caitlyn(BOTTOM,T200,pid9) (asiste ADC aliado)
- min 7.72: Qiyana(MIDDLE,T100,pid3) mata a Senna(UTILITY,T200,pid10); assists: - (muere support aliado)

### Case 4: EUW1_7783266689 (Senna + Ashe)

Predicho=0.263, real=0.943, error=0.680. Draft: Garen/Nidalee/Leblanc/Ashe/Senna vs FiddleSticks/Briar/KogMaw/Twitch/Lulu.
Lectura: modelo infrapredice un roaming extremo; la etiqueta viene sobre todo de presencia fuera de bot; support muere 4 vez/veces antes de 12; ADC muere 3 vez/veces antes de 12; support participa en kills tempranas. KDA final botlane aliada: support 3/12/16, ADC 3/9/5.

Eventos tempranos relevantes:
- min 1.88: Twitch(BOTTOM,T100,pid4) mata a Ashe(BOTTOM,T200,pid9); assists: Lulu(UTILITY,T100,pid5) (muere ADC aliado)
- min 2.00: Nidalee(JUNGLE,T200,pid7) mata a Twitch(BOTTOM,T100,pid4); assists: Ashe(BOTTOM,T200,pid9); Senna(UTILITY,T200,pid10) (asiste support aliado; asiste ADC aliado)
- min 2.15: Senna(UTILITY,T200,pid10) mata a Lulu(UTILITY,T100,pid5); assists: Nidalee(JUNGLE,T200,pid7) (participa support aliado)
- min 4.56: Briar(JUNGLE,T100,pid2) mata a Senna(UTILITY,T200,pid10); assists: Lulu(UTILITY,T100,pid5) (muere support aliado)
- min 4.75: Ashe(BOTTOM,T200,pid9) mata a Briar(JUNGLE,T100,pid2); assists: Senna(UTILITY,T200,pid10) (asiste support aliado; participa ADC aliado)
- min 4.79: Twitch(BOTTOM,T100,pid4) mata a Ashe(BOTTOM,T200,pid9); assists: Lulu(UTILITY,T100,pid5) (muere ADC aliado)
- min 5.97: Senna(UTILITY,T200,pid10) mata a Briar(JUNGLE,T100,pid2); assists: Nidalee(JUNGLE,T200,pid7) (participa support aliado)
- min 6.34: Garen(TOP,T200,pid6) mata a FiddleSticks(TOP,T100,pid1); assists: Nidalee(JUNGLE,T200,pid7); Senna(UTILITY,T200,pid10) (asiste support aliado)

### Case 5: EUW1_7714775914 (Senna + Kaisa)

Predicho=0.304, real=0.971, error=0.667. Draft: DrMundo/Kayn/Irelia/Kaisa/Senna vs Cassiopeia/Naafiri/TwistedFate/Ezreal/Rell.
Lectura: modelo infrapredice un roaming extremo; la etiqueta parece consistente: fuera de bot y lejos del ADC casi toda la ventana; support muere 3 vez/veces antes de 12; ADC muere 2 vez/veces antes de 12; support participa en kills tempranas. KDA final botlane aliada: support 3/9/16, ADC 9/6/7.

Eventos tempranos relevantes:
- min 1.22: Ezreal(BOTTOM,T100,pid4) mata a Kaisa(BOTTOM,T200,pid9); assists: Rell(UTILITY,T100,pid5) (muere ADC aliado)
- min 2.54: Rell(UTILITY,T100,pid5) mata a Senna(UTILITY,T200,pid10); assists: Ezreal(BOTTOM,T100,pid4) (muere support aliado)
- min 3.43: Kayn(JUNGLE,T200,pid7) mata a Rell(UTILITY,T100,pid5); assists: Irelia(MIDDLE,T200,pid8); Kaisa(BOTTOM,T200,pid9); Senna(UTILITY,T200,pid10) (asiste support aliado; asiste ADC aliado)
- min 5.18: Ezreal(BOTTOM,T100,pid4) mata a Kaisa(BOTTOM,T200,pid9); assists: Rell(UTILITY,T100,pid5) (muere ADC aliado)
- min 5.69: Irelia(MIDDLE,T200,pid8) mata a TwistedFate(MIDDLE,T100,pid3); assists: Kayn(JUNGLE,T200,pid7); Senna(UTILITY,T200,pid10) (asiste support aliado)
- min 6.49: Kaisa(BOTTOM,T200,pid9) mata a Naafiri(JUNGLE,T100,pid2); assists: Irelia(MIDDLE,T200,pid8); Senna(UTILITY,T200,pid10) (asiste support aliado; participa ADC aliado)
- min 7.16: DrMundo(TOP,T200,pid6) mata a Cassiopeia(TOP,T100,pid1); assists: Kayn(JUNGLE,T200,pid7); Senna(UTILITY,T200,pid10) (asiste support aliado)
- min 7.64: Naafiri(JUNGLE,T100,pid2) mata a Senna(UTILITY,T200,pid10); assists: - (muere support aliado)

## How to use

Use cases marked `consistent_full_roam_label` plus clear early event context as examples of unpredictable in-game variance. Treat cases marked `low_valid_*` or `possible_adc_death_base_coop_artifact` as cautionary label-limit examples.
