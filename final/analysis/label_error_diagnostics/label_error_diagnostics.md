# Label Error Diagnostics

This report diagnoses the largest HistGBT test errors using the frame-level timeline that produced the v5 support roaming label.

## Diagnostic Counts

| label_diagnostic                      | cases |
| ------------------------------------- | ----- |
| consistent_full_roam_label            | 12    |
| mostly_outside_bot_context            | 3     |
| low_valid_coop_frames                 | 2     |
| mostly_far_from_adc                   | 1     |
| low_valid_support_frames              | 1     |
| possible_adc_death_base_coop_artifact | 1     |

## Top Error Components

| error_rank | ally_utility_champion_name | ally_bottom_champion_name | enemy_utility_champion_name | enemy_bottom_champion_name | prediction | actual | abs_error | outside_ratio_v5 | far_ratio_v5 | xp_gap_v5 | valid_support_frames_v5 | valid_coop_frames_v5 | label_diagnostic                      | timeline_plot                            |
| ---------- | -------------------------- | ------------------------- | --------------------------- | -------------------------- | ---------- | ------ | --------- | ---------------- | ------------ | --------- | ----------------------- | -------------------- | ------------------------------------- | ---------------------------------------- |
| 1          | Yuumi                      | Smolder                   | Pyke                        | Velkoz                     | 0.209      | 1.000  | 0.791     | 1.000            | 1.000        | 1.000     | 5                       | 5.000                | consistent_full_roam_label            | timeline_case_01_EUW1_7831489390_200.png |
| 2          | Yuumi                      | Zeri                      | Sona                        | Lucian                     | 0.174      | 0.930  | 0.756     | 0.833            | 1.000        | 0.911     | 6                       | 5.000                | mostly_outside_bot_context            | timeline_case_02_EUW1_7706461344_100.png |
| 3          | Senna                      | Caitlyn                   | Blitzcrank                  | Tristana                   | 0.310      | 1.000  | 0.690     | 1.000            | 1.000        | 1.000     | 4                       | 4.000                | consistent_full_roam_label            | timeline_case_03_EUW1_7708715292_200.png |
| 4          | Senna                      | Ashe                      | Lulu                        | Twitch                     | 0.263      | 0.943  | 0.680     | 0.833            | 1.000        | 1.000     | 6                       | 5.000                | mostly_outside_bot_context            | timeline_case_04_EUW1_7783266689_200.png |
| 5          | Senna                      | Kaisa                     | Rell                        | Ezreal                     | 0.304      | 0.971  | 0.667     | 1.000            | 1.000        | 0.806     | 4                       | 4.000                | consistent_full_roam_label            | timeline_case_05_EUW1_7714775914_200.png |
| 6          | Karma                      | Caitlyn                   | Lulu                        | Ashe                       | 0.337      | 1.000  | 0.663     | 1.000            | 1.000        | 1.000     | 5                       | 4.000                | consistent_full_roam_label            | timeline_case_06_EUW1_7705744974_100.png |
| 7          | Sona                       | Ashe                      | Blitzcrank                  | Jinx                       | 0.341      | 1.000  | 0.659     | 1.000            | 1.000        | 1.000     | 7                       | 5.000                | consistent_full_roam_label            | timeline_case_07_EUW1_7830670024_100.png |
| 8          | Yuumi                      | Aphelios                  | Velkoz                      | Caitlyn                    | 0.158      | 0.817  | 0.658     | 0.667            | 0.833        | 0.859     | 6                       | 6.000                | mostly_far_from_adc                   | timeline_case_08_EUW1_7705182186_200.png |
| 9          | Velkoz                     | Twitch                    | Nami                        | Mel                        | 0.342      | 1.000  | 0.658     | 1.000            | 1.000        | 1.000     | 5                       | 5.000                | consistent_full_roam_label            | timeline_case_09_EUW1_7716308181_100.png |
| 10         | Lux                        | Caitlyn                   | Karma                       | Jhin                       | 0.343      | 1.000  | 0.657     | 1.000            | 1.000        | 1.000     | 3                       | 2.000                | low_valid_support_frames              | timeline_case_10_EUW1_7708270762_200.png |
| 11         | Nami                       | Yasuo                     | Soraka                      | Smolder                    | 0.320      | 0.977  | 0.657     | 1.000            | 1.000        | 0.846     | 6                       | 6.000                | consistent_full_roam_label            | timeline_case_11_EUW1_7830110048_100.png |
| 12         | Sona                       | Yunara                    | Camille                     | Corki                      | 0.347      | 0.998  | 0.651     | 1.000            | 1.000        | 0.988     | 6                       | 3.000                | low_valid_coop_frames                 | timeline_case_12_EUW1_7806861902_200.png |
| 13         | Yuumi                      | Tristana                  | Milio                       | Lucian                     | 0.131      | 0.777  | 0.647     | 0.571            | 0.750        | 0.975     | 7                       | 4.000                | possible_adc_death_base_coop_artifact | timeline_case_13_EUW1_7708834871_200.png |
| 14         | Braum                      | Vayne                     | Karma                       | Varus                      | 0.345      | 0.992  | 0.646     | 1.000            | 1.000        | 0.944     | 6                       | 5.000                | consistent_full_roam_label            | timeline_case_14_EUW1_7705296763_200.png |
| 15         | Taric                      | Lucian                    | Yuumi                       | Zeri                       | 0.269      | 0.913  | 0.643     | 0.833            | 1.000        | 0.802     | 6                       | 5.000                | mostly_outside_bot_context            | timeline_case_15_EUW1_7831144335_200.png |
| 16         | Senna                      | Veigar                    | Blitzcrank                  | Xerath                     | 0.358      | 1.000  | 0.642     | 1.000            | 1.000        | 1.000     | 6                       | 6.000                | consistent_full_roam_label            | timeline_case_16_EUW1_7719808217_100.png |
| 17         | Ivern                      | Lucian                    | Senna                       | Jhin                       | 0.362      | 1.000  | 0.638     | 1.000            | 1.000        | 1.000     | 7                       | 6.000                | consistent_full_roam_label            | timeline_case_17_EUW1_7765827606_200.png |
| 18         | Braum                      | Varus                     | Nautilus                    | Aphelios                   | 0.360      | 0.996  | 0.637     | 1.000            | 1.000        | 0.975     | 5                       | 5.000                | consistent_full_roam_label            | timeline_case_18_EUW1_7804333236_200.png |
| 19         | Yuumi                      | Corki                     | Sona                        | Zed                        | 0.219      | 0.855  | 0.637     | 0.800            | 1.000        | 0.510     | 5                       | 3.000                | low_valid_coop_frames                 | timeline_case_19_EUW1_7803978812_100.png |
| 20         | Zyra                       | Ashe                      | Thresh                      | Jinx                       | 0.289      | 0.922  | 0.634     | 1.000            | 1.000        | 0.490     | 4                       | 4.000                | consistent_full_roam_label            | timeline_case_20_EUW1_7831462570_200.png |

## Reading

- `outside_ratio_v5` is the share of valid support frames outside bot context.
- `far_ratio_v5` is the share of valid cooperation frames with support at least 2500 units away from ADC.
- `xp_gap_v5` increases when support XP lags behind ADC XP at the end of the window.
- Cases marked `consistent_full_roam_label` are likely real label extremes, not obvious scoring artifacts.
- Cases marked `low_valid_*` or `possible_adc_death_base_coop_artifact` deserve manual review before using as examples.
