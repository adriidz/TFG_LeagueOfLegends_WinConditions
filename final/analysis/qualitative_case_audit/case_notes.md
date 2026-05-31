# Qualitative Case Audit

This report consolidates model errors, label components, frame-level positions, map plots, and raw Riot timeline evidence.

## Top Errors: donde falla el modelo

| case_group | case_rank | match_id        | team_id | side | patch | ally_utility_champion_name | ally_bottom_champion_name | enemy_utility_champion_name | enemy_bottom_champion_name | ally_support_expert_score | ally_support_expert_archetype | ally_support_champion_mean_score | prediction | actual | abs_error | outside_ratio_v5 | far_ratio_v5 | support_early_deaths | adc_early_deaths | evidence_tag              |
| ---------- | --------- | --------------- | ------- | ---- | ----- | -------------------------- | ------------------------- | --------------------------- | -------------------------- | ------------------------- | ----------------------------- | -------------------------------- | ---------- | ------ | --------- | ---------------- | ------------ | -------------------- | ---------------- | ------------------------- |
| top_error  | 1         | EUW1_7831489390 | 200     | red  | 16.8  | Yuumi                      | Smolder                   | Pyke                        | Velkoz                     | 0.080                     | adc_enabler                   | 0.149                            | 0.209      | 1.000  | 0.791     | 1.000            | 1.000        | 4                    | 7                | chaotic_early_game        |
| top_error  | 2         | EUW1_7706461344 | 100     | blue | 16.2  | Yuumi                      | Zeri                      | Sona                        | Lucian                     | 0.080                     | adc_enabler                   | 0.149                            | 0.174      | 0.930  | 0.756     | 0.833            | 1.000        | 3                    | 6                | chaotic_early_game        |
| top_error  | 3         | EUW1_7708715292 | 200     | red  | 16.2  | Senna                      | Caitlyn                   | Blitzcrank                  | Tristana                   | 0.400                     | marksman_support              | 0.309                            | 0.310      | 1.000  | 0.690     | 1.000            | 1.000        | 6                    | 1                | chaotic_early_game        |
| top_error  | 4         | EUW1_7783266689 | 200     | red  | 16.5  | Senna                      | Ashe                      | Lulu                        | Twitch                     | 0.400                     | marksman_support              | 0.309                            | 0.263      | 0.943  | 0.680     | 0.833            | 1.000        | 4                    | 3                | chaotic_early_game        |
| top_error  | 5         | EUW1_7714775914 | 200     | red  | 16.2  | Senna                      | Kaisa                     | Rell                        | Ezreal                     | 0.400                     | marksman_support              | 0.309                            | 0.304      | 0.971  | 0.667     | 1.000            | 1.000        | 3                    | 2                | chaotic_early_game        |
| top_error  | 6         | EUW1_7705744974 | 100     | blue | 16.2  | Karma                      | Caitlyn                   | Lulu                        | Ashe                       | 0.420                     | lane_enchanter                | 0.387                            | 0.337      | 1.000  | 0.663     | 1.000            | 1.000        | 3                    | 3                | chaotic_early_game        |
| top_error  | 7         | EUW1_7830670024 | 100     | blue | 16.8  | Sona                       | Ashe                      | Blitzcrank                  | Jinx                       | 0.220                     | scaling_enchanter             | 0.334                            | 0.341      | 1.000  | 0.659     | 1.000            | 1.000        | 4                    | 4                | chaotic_early_game        |
| top_error  | 8         | EUW1_7705182186 | 200     | red  | 16.2  | Yuumi                      | Aphelios                  | Velkoz                      | Caitlyn                    | 0.080                     | adc_enabler                   | 0.149                            | 0.158      | 0.817  | 0.658     | 0.667            | 0.833        | 2                    | 5                | chaotic_early_game        |
| top_error  | 9         | EUW1_7716308181 | 100     | blue | 16.3  | Velkoz                     | Twitch                    | Nami                        | Mel                        | 0.280                     | lane_mage                     | 0.331                            | 0.342      | 1.000  | 0.658     | 1.000            | 1.000        | 3                    | 0                | clean_roam_like_candidate |
| top_error  | 10        | EUW1_7708270762 | 200     | red  | 16.2  | Lux                        | Caitlyn                   | Karma                       | Jhin                       | 0.320                     | lane_mage                     | 0.356                            | 0.343      | 1.000  | 0.657     | 1.000            | 1.000        | 4                    | 3                | label_quality_caution     |
| top_error  | 11        | EUW1_7830110048 | 100     | blue | 16.8  | Nami                       | Yasuo                     | Soraka                      | Smolder                    | 0.380                     | lane_enchanter                | 0.346                            | 0.320      | 0.977  | 0.657     | 1.000            | 1.000        | 2                    | 5                | chaotic_early_game        |
| top_error  | 12        | EUW1_7806861902 | 200     | red  | 16.7  | Sona                       | Yunara                    | Camille                     | Corki                      | 0.220                     | scaling_enchanter             | 0.334                            | 0.347      | 0.998  | 0.651     | 1.000            | 1.000        | 5                    | 1                | chaotic_early_game        |
| top_error  | 13        | EUW1_7708834871 | 200     | red  | 16.2  | Yuumi                      | Tristana                  | Milio                       | Lucian                     | 0.080                     | adc_enabler                   | 0.149                            | 0.131      | 0.777  | 0.647     | 0.571            | 0.750        | 1                    | 4                | chaotic_early_game        |
| top_error  | 14        | EUW1_7705296763 | 200     | red  | 16.2  | Braum                      | Vayne                     | Karma                       | Varus                      | 0.480                     | protective_warden             | 0.359                            | 0.345      | 0.992  | 0.646     | 1.000            | 1.000        | 3                    | 3                | chaotic_early_game        |
| top_error  | 15        | EUW1_7831144335 | 200     | red  | 16.8  | Taric                      | Lucian                    | Yuumi                       | Zeri                       | 0.300                     | protective_warden             | 0.322                            | 0.269      | 0.913  | 0.643     | 0.833            | 1.000        | 1                    | 1                | clean_roam_like_candidate |
| top_error  | 16        | EUW1_7719808217 | 100     | blue | 16.3  | Senna                      | Veigar                    | Blitzcrank                  | Xerath                     | 0.400                     | marksman_support              | 0.309                            | 0.358      | 1.000  | 0.642     | 1.000            | 1.000        | 4                    | 2                | chaotic_early_game        |
| top_error  | 17        | EUW1_7765827606 | 200     | red  | 16.5  | Ivern                      | Lucian                    | Senna                       | Jhin                       |                           |                               | 0.391                            | 0.362      | 1.000  | 0.638     | 1.000            | 1.000        | 5                    | 2                | chaotic_early_game        |
| top_error  | 18        | EUW1_7804333236 | 200     | red  | 16.6  | Braum                      | Varus                     | Nautilus                    | Aphelios                   | 0.480                     | protective_warden             | 0.359                            | 0.360      | 0.996  | 0.637     | 1.000            | 1.000        | 2                    | 3                | chaotic_early_game        |
| top_error  | 19        | EUW1_7803978812 | 100     | blue | 16.6  | Yuumi                      | Corki                     | Sona                        | Zed                        | 0.080                     | adc_enabler                   | 0.149                            | 0.219      | 0.855  | 0.637     | 0.800            | 1.000        | 2                    | 6                | chaotic_early_game        |
| top_error  | 20        | EUW1_7831462570 | 200     | red  | 16.8  | Zyra                       | Ashe                      | Thresh                      | Jinx                       | 0.340                     | lane_mage                     | 0.306                            | 0.289      | 0.922  | 0.634     | 1.000            | 1.000        | 3                    | 8                | chaotic_early_game        |

## Bottom Errors: cuando el modelo acierta

| case_group   | case_rank | match_id        | team_id | side | patch | ally_utility_champion_name | ally_bottom_champion_name | enemy_utility_champion_name | enemy_bottom_champion_name | ally_support_expert_score | ally_support_expert_archetype | ally_support_champion_mean_score | prediction | actual | abs_error | outside_ratio_v5 | far_ratio_v5 | support_early_deaths | adc_early_deaths | evidence_tag       |
| ------------ | --------- | --------------- | ------- | ---- | ----- | -------------------------- | ------------------------- | --------------------------- | -------------------------- | ------------------------- | ----------------------------- | -------------------------------- | ---------- | ------ | --------- | ---------------- | ------------ | -------------------- | ---------------- | ------------------ |
| bottom_error | 1         | EUW1_7739311514 | 200     | red  | 16.3  | Yuumi                      | Lucian                    | Senna                       | Zilean                     | 0.080                     | adc_enabler                   | 0.149                            | 0.153      | 0.153  | 0.000     | 0.143            | 0.000        | 1                    | 1                | accurate_low       |
| bottom_error | 2         | EUW1_7827297324 | 100     | blue | 16.8  | Yuumi                      | Twitch                    | Rell                        | Vayne                      | 0.080                     | adc_enabler                   | 0.149                            | 0.164      | 0.164  | 0.000     | 0.000            | 0.000        | 6                    | 5                | chaotic_early_game |
| bottom_error | 3         | EUW1_7737055177 | 200     | red  | 16.3  | Yuumi                      | Vayne                     | Lux                         | Jhin                       | 0.080                     | adc_enabler                   | 0.149                            | 0.155      | 0.155  | 0.000     | 0.143            | 0.000        | 2                    | 2                | accurate_low       |
| bottom_error | 4         | EUW1_7753025529 | 100     | blue | 16.4  | Senna                      | Nilah                     | Braum                       | Smolder                    | 0.400                     | marksman_support              | 0.309                            | 0.247      | 0.246  | 0.001     | 0.167            | 0.167        | 4                    | 3                | chaotic_early_game |
| bottom_error | 5         | EUW1_7740461495 | 100     | blue | 16.3  | Yuumi                      | Ezreal                    | Senna                       | Nilah                      | 0.080                     | adc_enabler                   | 0.149                            | 0.146      | 0.145  | 0.001     | 0.000            | 0.000        | 0                    | 0                | accurate_low       |
| bottom_error | 6         | EUW1_7717169345 | 200     | red  | 16.3  | Lulu                       | Varus                     | Braum                       | Hwei                       | 0.280                     | adc_enabler                   | 0.322                            | 0.329      | 0.329  | 0.000     | 0.143            | 0.143        | 1                    | 1                | accurate_mid       |
| bottom_error | 7         | EUW1_7711409386 | 100     | blue | 16.2  | Senna                      | Lucian                    | Nami                        | Draven                     | 0.400                     | marksman_support              | 0.309                            | 0.277      | 0.277  | 0.000     | 0.000            | 0.200        | 2                    | 3                | chaotic_early_game |
| bottom_error | 8         | EUW1_7822907738 | 100     | blue | 16.8  | Sona                       | Velkoz                    | Rakan                       | Yasuo                      | 0.220                     | scaling_enchanter             | 0.334                            | 0.368      | 0.368  | 0.000     | 0.000            | 0.200        | 2                    | 2                | accurate_mid       |
| bottom_error | 9         | EUW1_7760148189 | 200     | red  | 16.4  | Nautilus                   | Kaisa                     | Bard                        | Jinx                       | 0.840                     | engage_roamer                 | 0.416                            | 0.470      | 0.470  | 0.000     | 0.143            | 0.429        | 2                    | 1                | chaotic_early_game |
| bottom_error | 10        | EUW1_7747844563 | 200     | red  | 16.4  | Leona                      | MissFortune               | Janna                       | Senna                      | 0.800                     | engage_roamer                 | 0.414                            | 0.409      | 0.409  | 0.000     | 0.286            | 0.200        | 1                    | 2                | accurate_mid       |
| bottom_error | 11        | EUW1_7728913380 | 200     | red  | 16.3  | Alistar                    | Ekko                      | Karma                       | Veigar                     | 0.820                     | engage_roamer                 | 0.450                            | 0.522      | 0.522  | 0.000     | 0.429            | 0.286        | 3                    | 1                | accurate_mid       |
| bottom_error | 12        | EUW1_7758367754 | 100     | blue | 16.4  | Elise                      | Jhin                      | Braum                       | Smolder                    | 0.720                     | pick_roamer                   | 0.534                            | 0.502      | 0.502  | 0.000     | 0.286            | 0.286        | 1                    | 0                | accurate_mid       |
| bottom_error | 13        | EUW1_7808656114 | 100     | blue | 16.7  | Bard                       | Seraphine                 | Janna                       | Yasuo                      | 0.950                     | roaming_specialist            | 0.506                            | 0.539      | 0.540  | 0.000     | 0.143            | 0.500        | 1                    | 1                | accurate_mid       |
| bottom_error | 14        | EUW1_7753745474 | 200     | red  | 16.4  | Shaco                      | Caitlyn                   | Bard                        | Jhin                       | 0.740                     | ambush_roamer                 | 0.540                            | 0.572      | 0.572  | 0.000     | 0.286            | 0.750        | 1                    | 5                | chaotic_early_game |
| bottom_error | 15        | EUW1_7748838971 | 200     | red  | 16.4  | Rell                       | Seraphine                 | Shaco                       | AurelionSol                | 0.820                     | engage_roamer                 | 0.437                            | 0.563      | 0.562  | 0.000     | 0.143            | 0.571        | 2                    | 1                | accurate_mid       |
| bottom_error | 16        | EUW1_7712893387 | 100     | blue | 16.2  | AurelionSol                | Jhin                      | Leona                       | KogMaw                     |                           |                               | 0.670                            | 0.665      | 0.780  | 0.115     | 0.571            | 0.857        | 1                    | 1                | accurate_high      |
| bottom_error | 17        | EUW1_7827524842 | 100     | blue | 16.8  | Shaco                      | AurelionSol               | Alistar                     | Veigar                     | 0.740                     | ambush_roamer                 | 0.540                            | 0.627      | 0.757  | 0.130     | 0.571            | 0.667        | 1                    | 2                | chaotic_early_game |
| bottom_error | 18        | EUW1_7736622895 | 100     | blue | 16.3  | Bard                       | Sivir                     | Nautilus                    | Seraphine                  | 0.950                     | roaming_specialist            | 0.506                            | 0.614      | 0.765  | 0.151     | 0.571            | 0.714        | 0                    | 0                | accurate_high      |
| bottom_error | 19        | EUW1_7772199297 | 100     | blue | 16.5  | Bard                       | Ziggs                     | Karma                       | Ezreal                     | 0.950                     | roaming_specialist            | 0.506                            | 0.607      | 0.759  | 0.153     | 0.429            | 0.857        | 1                    | 0                | accurate_high      |
| bottom_error | 20        | EUW1_7761485723 | 200     | red  | 16.4  | Elise                      | Ezreal                    | Pantheon                    | Seraphine                  | 0.720                     | pick_roamer                   | 0.534                            | 0.586      | 0.755  | 0.169     | 0.500            | 0.750        | 4                    | 1                | chaotic_early_game |

## Patrones encontrados

| evidence_tag              | cases |
| ------------------------- | ----- |
| chaotic_early_game        | 24    |
| accurate_mid              | 7     |
| accurate_low              | 3     |
| accurate_high             | 3     |
| clean_roam_like_candidate | 2     |
| label_quality_caution     | 1     |

## Limitaciones de etiqueta

- `support_roam_score` debe leerse como `roam-like displacement` o separacion support-ADC, no como intencion tactica garantizada.
- Casos con muchas muertes tempranas o pocos frames validos deben usarse como cautela metodologica.
- Los mapas cronologicos en `case_plots/` permiten auditar si las posiciones y zonas geometricas parecen correctas.

## Casos recomendados para la memoria

### top_error #1: EUW1_7831489390 T200

Draft: Sion/Talon/KogMaw/Smolder/Yuumi vs Jayce/Shaco/Azir/Velkoz/Pyke.
Expert expected support score: ally Yuumi=0.080 (adc_enabler), enemy Pyke=0.920 (roaming_assassin).
Empirical champion mean: ally Yuumi=0.149, n=4689, enemy Pyke=0.492, n=11462.
Prediccion=0.209, actual=1.000, abs_error=0.791, tag=chaotic_early_game.
Mapa: `case_plots/top_error_01_EUW1_7831489390_200_map.png`. Timeline frame-level: `case_plots/top_error_01_EUW1_7831489390_200_timeline.png`.

Eventos tempranos relevantes:
- min 1.36: CHAMPION_KILL | Velkoz(BOTTOM,T100,pid4) -> Yuumi(UTILITY,T200,pid10) | assists: Pyke(UTILITY,T100,pid5) (support died)
- min 1.39: CHAMPION_KILL | Smolder(BOTTOM,T200,pid9) -> Pyke(UTILITY,T100,pid5) | assists: Yuumi(UTILITY,T200,pid10) (support assist)
- min 1.48: CHAMPION_KILL | Velkoz(BOTTOM,T100,pid4) -> Smolder(BOTTOM,T200,pid9) | assists: Pyke(UTILITY,T100,pid5) (ADC died)
- min 3.03: CHAMPION_KILL | Shaco(JUNGLE,T100,pid2) -> Smolder(BOTTOM,T200,pid9) | assists: Velkoz(BOTTOM,T100,pid4); Pyke(UTILITY,T100,pid5) (ADC died)
- min 3.87: CHAMPION_KILL | Velkoz(BOTTOM,T100,pid4) -> Smolder(BOTTOM,T200,pid9) | assists: Shaco(JUNGLE,T100,pid2); Pyke(UTILITY,T100,pid5) (ADC died)
- min 3.99: CHAMPION_KILL | Velkoz(BOTTOM,T100,pid4) -> Yuumi(UTILITY,T200,pid10) | assists: Pyke(UTILITY,T100,pid5) (support died)
- min 4.66: CHAMPION_KILL | Yuumi(UTILITY,T200,pid10) -> Shaco(JUNGLE,T100,pid2) | assists: Sion(TOP,T200,pid6); Talon(JUNGLE,T200,pid7)
- min 6.65: CHAMPION_KILL | Velkoz(BOTTOM,T100,pid4) -> Smolder(BOTTOM,T200,pid9) (ADC died)

### top_error #2: EUW1_7706461344 T100

Draft: KSante/Viego/Cassiopeia/Zeri/Yuumi vs Camille/Graves/TwistedFate/Lucian/Sona.
Expert expected support score: ally Yuumi=0.080 (adc_enabler), enemy Sona=0.220 (scaling_enchanter).
Empirical champion mean: ally Yuumi=0.149, n=4689, enemy Sona=0.334, n=11424.
Prediccion=0.174, actual=0.930, abs_error=0.756, tag=chaotic_early_game.
Mapa: `case_plots/top_error_02_EUW1_7706461344_100_map.png`. Timeline frame-level: `case_plots/top_error_02_EUW1_7706461344_100_timeline.png`.

Eventos tempranos relevantes:
- min 1.81: CHAMPION_KILL | Sona(UTILITY,T200,pid10) -> Zeri(BOTTOM,T100,pid4) | assists: Lucian(BOTTOM,T200,pid9) (ADC died)
- min 3.49: CHAMPION_KILL | Sona(UTILITY,T200,pid10) -> Zeri(BOTTOM,T100,pid4) | assists: Lucian(BOTTOM,T200,pid9) (ADC died)
- min 4.34: CHAMPION_KILL | Lucian(BOTTOM,T200,pid9) -> Zeri(BOTTOM,T100,pid4) | assists: Sona(UTILITY,T200,pid10) (ADC died)
- min 5.70: CHAMPION_KILL | Lucian(BOTTOM,T200,pid9) -> Yuumi(UTILITY,T100,pid5) | assists: Sona(UTILITY,T200,pid10) (support died)
- min 6.19: CHAMPION_KILL | Lucian(BOTTOM,T200,pid9) -> Zeri(BOTTOM,T100,pid4) | assists: Sona(UTILITY,T200,pid10) (ADC died)
- min 7.47: CHAMPION_KILL | Graves(JUNGLE,T200,pid7) -> Yuumi(UTILITY,T100,pid5) (support died)
- min 7.91: CHAMPION_KILL | Graves(JUNGLE,T200,pid7) -> Zeri(BOTTOM,T100,pid4) | assists: Lucian(BOTTOM,T200,pid9); Sona(UTILITY,T200,pid10) (ADC died)
- min 9.27: CHAMPION_KILL | Graves(JUNGLE,T200,pid7) -> Zeri(BOTTOM,T100,pid4) | assists: TwistedFate(MIDDLE,T200,pid8); Sona(UTILITY,T200,pid10) (ADC died)

### top_error #3: EUW1_7708715292 T200

Draft: Teemo/Diana/Yasuo/Caitlyn/Senna vs Swain/Nidalee/Qiyana/Tristana/Blitzcrank.
Expert expected support score: ally Senna=0.400 (marksman_support), enemy Blitzcrank=0.780 (pick_roamer).
Empirical champion mean: ally Senna=0.309, n=10154, enemy Blitzcrank=0.435, n=8649.
Prediccion=0.310, actual=1.000, abs_error=0.690, tag=chaotic_early_game.
Mapa: `case_plots/top_error_03_EUW1_7708715292_200_map.png`. Timeline frame-level: `case_plots/top_error_03_EUW1_7708715292_200_timeline.png`.

Eventos tempranos relevantes:
- min 2.63: CHAMPION_KILL | Tristana(BOTTOM,T100,pid4) -> Senna(UTILITY,T200,pid10) | assists: Blitzcrank(UTILITY,T100,pid5) (support died)
- min 2.71: CHAMPION_KILL | Caitlyn(BOTTOM,T200,pid9) -> Tristana(BOTTOM,T100,pid4) | assists: Senna(UTILITY,T200,pid10) (support assist)
- min 4.21: CHAMPION_KILL | Tristana(BOTTOM,T100,pid4) -> Caitlyn(BOTTOM,T200,pid9) | assists: Blitzcrank(UTILITY,T100,pid5) (ADC died)
- min 4.51: CHAMPION_KILL | Tristana(BOTTOM,T100,pid4) -> Senna(UTILITY,T200,pid10) | assists: Blitzcrank(UTILITY,T100,pid5) (support died)
- min 5.94: CHAMPION_KILL | Diana(JUNGLE,T200,pid7) -> Blitzcrank(UTILITY,T100,pid5) | assists: Caitlyn(BOTTOM,T200,pid9) (ADC assist)
- min 5.96: CHAMPION_KILL | Qiyana(MIDDLE,T100,pid3) -> Senna(UTILITY,T200,pid10) (support died)
- min 6.18: CHAMPION_KILL | Diana(JUNGLE,T200,pid7) -> Tristana(BOTTOM,T100,pid4) | assists: Caitlyn(BOTTOM,T200,pid9) (ADC assist)
- min 7.72: CHAMPION_KILL | Qiyana(MIDDLE,T100,pid3) -> Senna(UTILITY,T200,pid10) (support died)

### top_error #4: EUW1_7783266689 T200

Draft: Garen/Nidalee/Leblanc/Ashe/Senna vs FiddleSticks/Briar/KogMaw/Twitch/Lulu.
Expert expected support score: ally Senna=0.400 (marksman_support), enemy Lulu=0.280 (adc_enabler).
Empirical champion mean: ally Senna=0.309, n=10154, enemy Lulu=0.322, n=22250.
Prediccion=0.263, actual=0.943, abs_error=0.680, tag=chaotic_early_game.
Mapa: `case_plots/top_error_04_EUW1_7783266689_200_map.png`. Timeline frame-level: `case_plots/top_error_04_EUW1_7783266689_200_timeline.png`.

Eventos tempranos relevantes:
- min 1.88: CHAMPION_KILL | Twitch(BOTTOM,T100,pid4) -> Ashe(BOTTOM,T200,pid9) | assists: Lulu(UTILITY,T100,pid5) (ADC died)
- min 2.00: CHAMPION_KILL | Nidalee(JUNGLE,T200,pid7) -> Twitch(BOTTOM,T100,pid4) | assists: Ashe(BOTTOM,T200,pid9); Senna(UTILITY,T200,pid10) (support assist; ADC assist)
- min 2.15: CHAMPION_KILL | Senna(UTILITY,T200,pid10) -> Lulu(UTILITY,T100,pid5) | assists: Nidalee(JUNGLE,T200,pid7)
- min 4.37: TURRET_PLATE_DESTROYED | Senna(UTILITY,T200,pid10) -> 
- min 4.56: CHAMPION_KILL | Briar(JUNGLE,T100,pid2) -> Senna(UTILITY,T200,pid10) | assists: Lulu(UTILITY,T100,pid5) (support died)
- min 4.75: CHAMPION_KILL | Ashe(BOTTOM,T200,pid9) -> Briar(JUNGLE,T100,pid2) | assists: Senna(UTILITY,T200,pid10) (support assist)
- min 4.79: CHAMPION_KILL | Twitch(BOTTOM,T100,pid4) -> Ashe(BOTTOM,T200,pid9) | assists: Lulu(UTILITY,T100,pid5) (ADC died)
- min 5.97: CHAMPION_KILL | Senna(UTILITY,T200,pid10) -> Briar(JUNGLE,T100,pid2) | assists: Nidalee(JUNGLE,T200,pid7)

### top_error #5: EUW1_7714775914 T200

Draft: DrMundo/Kayn/Irelia/Kaisa/Senna vs Cassiopeia/Naafiri/TwistedFate/Ezreal/Rell.
Expert expected support score: ally Senna=0.400 (marksman_support), enemy Rell=0.820 (engage_roamer).
Empirical champion mean: ally Senna=0.309, n=10154, enemy Rell=0.437, n=12474.
Prediccion=0.304, actual=0.971, abs_error=0.667, tag=chaotic_early_game.
Mapa: `case_plots/top_error_05_EUW1_7714775914_200_map.png`. Timeline frame-level: `case_plots/top_error_05_EUW1_7714775914_200_timeline.png`.

Eventos tempranos relevantes:
- min 1.22: CHAMPION_KILL | Ezreal(BOTTOM,T100,pid4) -> Kaisa(BOTTOM,T200,pid9) | assists: Rell(UTILITY,T100,pid5) (ADC died)
- min 2.54: CHAMPION_KILL | Rell(UTILITY,T100,pid5) -> Senna(UTILITY,T200,pid10) | assists: Ezreal(BOTTOM,T100,pid4) (support died)
- min 3.43: CHAMPION_KILL | Kayn(JUNGLE,T200,pid7) -> Rell(UTILITY,T100,pid5) | assists: Irelia(MIDDLE,T200,pid8); Kaisa(BOTTOM,T200,pid9); Senna(UTILITY,T200,pid10) (support assist; ADC assist)
- min 4.19: TURRET_PLATE_DESTROYED | Senna(UTILITY,T200,pid10) -> 
- min 5.18: CHAMPION_KILL | Ezreal(BOTTOM,T100,pid4) -> Kaisa(BOTTOM,T200,pid9) | assists: Rell(UTILITY,T100,pid5) (ADC died)
- min 5.69: CHAMPION_KILL | Irelia(MIDDLE,T200,pid8) -> TwistedFate(MIDDLE,T100,pid3) | assists: Kayn(JUNGLE,T200,pid7); Senna(UTILITY,T200,pid10) (support assist)
- min 6.49: CHAMPION_KILL | Kaisa(BOTTOM,T200,pid9) -> Naafiri(JUNGLE,T100,pid2) | assists: Irelia(MIDDLE,T200,pid8); Senna(UTILITY,T200,pid10) (support assist)
- min 7.16: CHAMPION_KILL | DrMundo(TOP,T200,pid6) -> Cassiopeia(TOP,T100,pid1) | assists: Kayn(JUNGLE,T200,pid7); Senna(UTILITY,T200,pid10) (support assist)

### bottom_error #16: EUW1_7712893387 T100

Draft: Jayce/Diana/Varus/Jhin/AurelionSol vs Teemo/Hecarim/Yone/KogMaw/Leona.
Expert expected support score: ally AurelionSol=NA, enemy Leona=0.800 (engage_roamer).
Empirical champion mean: ally AurelionSol=0.670, n=136, enemy Leona=0.414, n=12027.
Prediccion=0.665, actual=0.780, abs_error=0.115, tag=accurate_high.
Mapa: `case_plots/bottom_error_16_EUW1_7712893387_100_map.png`. Timeline frame-level: `case_plots/bottom_error_16_EUW1_7712893387_100_timeline.png`.

Eventos tempranos relevantes:
- min 5.47: CHAMPION_KILL | AurelionSol(UTILITY,T100,pid5) -> Hecarim(JUNGLE,T200,pid7) | assists: Diana(JUNGLE,T100,pid2)
- min 5.52: CHAMPION_KILL | KogMaw(BOTTOM,T200,pid9) -> AurelionSol(UTILITY,T100,pid5) | assists: Hecarim(JUNGLE,T200,pid7); Leona(UTILITY,T200,pid10) (support died)
- min 5.58: CHAMPION_KILL | Diana(JUNGLE,T100,pid2) -> KogMaw(BOTTOM,T200,pid9) | assists: Jhin(BOTTOM,T100,pid4) (ADC assist)
- min 5.76: ELITE_MONSTER_KILL | Yone(MIDDLE,T200,pid8) ->  | assists: Jhin(BOTTOM,T100,pid4); AurelionSol(UTILITY,T100,pid5); KogMaw(BOTTOM,T200,pid9); Leona(UTILITY,T200,pid10) (support assist; ADC assist)
- min 6.72: CHAMPION_KILL | Diana(JUNGLE,T100,pid2) -> KogMaw(BOTTOM,T200,pid9) | assists: AurelionSol(UTILITY,T100,pid5) (support assist)
- min 7.22: TURRET_PLATE_DESTROYED | Jhin(BOTTOM,T100,pid4) -> 
- min 8.67: CHAMPION_KILL | Varus(MIDDLE,T100,pid3) -> Yone(MIDDLE,T200,pid8) | assists: AurelionSol(UTILITY,T100,pid5) (support assist)
- min 9.68: CHAMPION_KILL | AurelionSol(UTILITY,T100,pid5) -> Yone(MIDDLE,T200,pid8) | assists: Varus(MIDDLE,T100,pid3)

### bottom_error #18: EUW1_7736622895 T100

Draft: Renekton/Ambessa/Katarina/Sivir/Bard vs Aurora/Zac/Yone/Seraphine/Nautilus.
Expert expected support score: ally Bard=0.950 (roaming_specialist), enemy Nautilus=0.840 (engage_roamer).
Empirical champion mean: ally Bard=0.506, n=22623, enemy Nautilus=0.416, n=28879.
Prediccion=0.614, actual=0.765, abs_error=0.151, tag=accurate_high.
Mapa: `case_plots/bottom_error_18_EUW1_7736622895_100_map.png`. Timeline frame-level: `case_plots/bottom_error_18_EUW1_7736622895_100_timeline.png`.

Eventos tempranos relevantes:
- min 5.67: CHAMPION_KILL | Bard(UTILITY,T100,pid5) -> Nautilus(UTILITY,T200,pid10) | assists: Ambessa(JUNGLE,T100,pid2); Sivir(BOTTOM,T100,pid4) (ADC assist)
- min 6.76: CHAMPION_KILL | Renekton(TOP,T100,pid1) -> Zac(JUNGLE,T200,pid7) | assists: Bard(UTILITY,T100,pid5) (support assist)
- min 8.36: CHAMPION_KILL | Katarina(MIDDLE,T100,pid3) -> Aurora(TOP,T200,pid6) | assists: Renekton(TOP,T100,pid1); Bard(UTILITY,T100,pid5) (support assist)
- min 8.71: CHAMPION_KILL | Katarina(MIDDLE,T100,pid3) -> Nautilus(UTILITY,T200,pid10) | assists: Ambessa(JUNGLE,T100,pid2); Bard(UTILITY,T100,pid5) (support assist)
- min 8.91: CHAMPION_KILL | Katarina(MIDDLE,T100,pid3) -> Yone(MIDDLE,T200,pid8) | assists: Ambessa(JUNGLE,T100,pid2); Bard(UTILITY,T100,pid5) (support assist)
- min 9.14: TURRET_PLATE_DESTROYED | Sivir(BOTTOM,T100,pid4) -> 
- min 10.27: TURRET_PLATE_DESTROYED | Sivir(BOTTOM,T100,pid4) -> 
- min 10.45: CHAMPION_KILL | Katarina(MIDDLE,T100,pid3) -> Nautilus(UTILITY,T200,pid10) | assists: Sivir(BOTTOM,T100,pid4); Bard(UTILITY,T100,pid5) (support assist; ADC assist)

### bottom_error #19: EUW1_7772199297 T100

Draft: Ambessa/Graves/Smolder/Ziggs/Bard vs Garen/XinZhao/Galio/Ezreal/Karma.
Expert expected support score: ally Bard=0.950 (roaming_specialist), enemy Karma=0.420 (lane_enchanter).
Empirical champion mean: ally Bard=0.506, n=22623, enemy Karma=0.387, n=27136.
Prediccion=0.607, actual=0.759, abs_error=0.153, tag=accurate_high.
Mapa: `case_plots/bottom_error_19_EUW1_7772199297_100_map.png`. Timeline frame-level: `case_plots/bottom_error_19_EUW1_7772199297_100_timeline.png`.

Eventos tempranos relevantes:
- min 7.31: CHAMPION_KILL | Ziggs(BOTTOM,T100,pid4) -> Ezreal(BOTTOM,T200,pid9) | assists: Bard(UTILITY,T100,pid5) (support assist)
- min 9.28: CHAMPION_KILL | Ziggs(BOTTOM,T100,pid4) -> Ezreal(BOTTOM,T200,pid9) | assists: Bard(UTILITY,T100,pid5) (support assist)
- min 9.61: CHAMPION_KILL | XinZhao(JUNGLE,T200,pid7) -> Bard(UTILITY,T100,pid5) | assists: Karma(UTILITY,T200,pid10) (support died)
