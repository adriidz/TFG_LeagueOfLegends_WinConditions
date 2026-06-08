# Empirical Ceiling / Repeatable Draft Signal

## Methodological note

ICC and R2 are not the same metric. ICC is reported here as a descriptive in-sample train statistic: it summarizes consistency within repeated draft groups after filtering small groups. The group-mean R2 below is the model-like reference: group means are fitted only on train, applied to test, and unseen test groups fall back to the train global mean. This OOS R2 is the value that can be compared with model test R2.

## Train ICC

| grouping | icc | n_groups | n_rows | mean_group_size | metric_role |
| --- | --- | --- | --- | --- | --- |
| support_champion | 0.1214 | 144 | 268289 | 1863.1181 | descriptive_in_sample_consistency |
| support_champion+side | 0.1212 | 255 | 268201 | 1051.7686 | descriptive_in_sample_consistency |
| botlane_champions | 0.1394 | 2518 | 262577 | 104.2800 | descriptive_in_sample_consistency |
| botlane_champions+side | 0.1391 | 3845 | 259164 | 67.4029 | descriptive_in_sample_consistency |
| sup_vs_enemy_sup_champion | 0.1316 | 2720 | 261959 | 96.3085 | descriptive_in_sample_consistency |
| support_riot_class | 0.0264 | 7 | 268322 | 38331.7143 | descriptive_in_sample_consistency |
| botlane_riot_classes | 0.0325 | 47 | 268319 | 5708.9149 | descriptive_in_sample_consistency |
| all_10_riot_classes | 0.0186 | 7225 | 88500 | 12.2491 | descriptive_in_sample_consistency |
| support_archetype | 0.0837 | 29 | 268322 | 9252.4828 | descriptive_in_sample_consistency |
| support_archetype+side | 0.0823 | 58 | 268322 | 4626.2414 | descriptive_in_sample_consistency |
| botlane_archetypes | 0.0932 | 388 | 267966 | 690.6340 | descriptive_in_sample_consistency |
| botlane_archetypes+side | 0.0931 | 635 | 267569 | 421.3685 | descriptive_in_sample_consistency |
| sup_vs_enemy_sup_archetype | 0.0901 | 385 | 267914 | 695.8805 | descriptive_in_sample_consistency |
| sup+jungle_archetypes | 0.0818 | 390 | 268019 | 687.2282 | descriptive_in_sample_consistency |
| sup+jungle_archetypes+side | 0.0819 | 653 | 267647 | 409.8729 | descriptive_in_sample_consistency |
| sup+jungle+top_archetypes | 0.0816 | 2650 | 262420 | 99.0264 | descriptive_in_sample_consistency |
| botlane_vs_enemy_bot_archetypes | 0.1006 | 4800 | 246287 | 51.3098 | descriptive_in_sample_consistency |
| ally_team_archetypes | 0.0769 | 10122 | 173051 | 17.0965 | descriptive_in_sample_consistency |
| ally_team_archetypes+side | 0.0711 | 10534 | 146119 | 13.8712 | descriptive_in_sample_consistency |
| all_10_archetypes | nan | 0 | 0 | nan | descriptive_in_sample_consistency |
| all_10_archetypes+side | nan | 0 | 0 | nan | descriptive_in_sample_consistency |

## Out-of-Sample Group-Mean R2

| grouping | r2_group_mean_oos | n_train_groups | n_test_groups | n_test_rows | n_unseen_test_groups | n_unseen_test_rows | unseen_test_row_rate | train_global_mean | group_means_fit_split | predicted_split |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| support_champion | 0.1249 | 164 | 149 | 57468 | 4 | 4 | 0.0001 | 0.3916 | train | test |
| support_champion+side | 0.1246 | 313 | 274 | 57468 | 9 | 10 | 0.0002 | 0.3916 | train | test |
| botlane_champions | 0.1239 | 5846 | 3447 | 57468 | 371 | 386 | 0.0067 | 0.3916 | train | test |
| botlane_champions+side | 0.1132 | 9284 | 5329 | 57468 | 601 | 636 | 0.0111 | 0.3916 | train | test |
| sup_vs_enemy_sup_champion | 0.1200 | 6330 | 3642 | 57468 | 388 | 414 | 0.0072 | 0.3916 | train | test |
| support_riot_class | 0.0218 | 7 | 7 | 57468 | 0 | 0 | 0.0000 | 0.3916 | train | test |
| botlane_riot_classes | 0.0301 | 49 | 46 | 57468 | 0 | 0 | 0.0000 | 0.3916 | train | test |
| all_10_riot_classes | -0.1824 | 149794 | 42476 | 57468 | 24556 | 25285 | 0.4400 | 0.3916 | train | test |
| support_archetype | 0.0852 | 29 | 29 | 57468 | 0 | 0 | 0.0000 | 0.3916 | train | test |
| support_archetype+side | 0.0852 | 58 | 58 | 57468 | 0 | 0 | 0.0000 | 0.3916 | train | test |
| botlane_archetypes | 0.0946 | 564 | 432 | 57468 | 12 | 13 | 0.0002 | 0.3916 | train | test |
| botlane_archetypes+side | 0.0930 | 1007 | 728 | 57468 | 28 | 29 | 0.0005 | 0.3916 | train | test |
| sup_vs_enemy_sup_archetype | 0.0944 | 573 | 433 | 57468 | 15 | 22 | 0.0004 | 0.3916 | train | test |
| sup+jungle_archetypes | 0.0841 | 541 | 423 | 57468 | 14 | 17 | 0.0003 | 0.3916 | train | test |
| sup+jungle_archetypes+side | 0.0822 | 986 | 737 | 57468 | 30 | 35 | 0.0006 | 0.3916 | train | test |
| sup+jungle+top_archetypes | 0.0651 | 5913 | 3502 | 57468 | 340 | 376 | 0.0065 | 0.3916 | train | test |
| botlane_vs_enemy_bot_archetypes | 0.0617 | 18737 | 8425 | 57468 | 1738 | 1850 | 0.0322 | 0.3916 | train | test |
| ally_team_archetypes | -0.0787 | 76587 | 27271 | 57468 | 9741 | 10226 | 0.1779 | 0.3916 | train | test |
| ally_team_archetypes+side | -0.1160 | 97992 | 32950 | 57468 | 13225 | 13812 | 0.2403 | 0.3916 | train | test |
| all_10_archetypes | -0.0023 | 267611 | 57432 | 57468 | 57124 | 57158 | 0.9946 | 0.3916 | train | test |
| all_10_archetypes+side | -0.0017 | 267928 | 57452 | 57468 | 57302 | 57318 | 0.9974 | 0.3916 | train | test |

## Combined View

| grouping | columns | icc_train | r2_group_mean_oos | n_train_groups_icc_min_size | n_train_groups_oos_means | n_test_groups | n_test_rows | n_unseen_test_groups | n_unseen_test_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| support_champion | ['ally_utility_champion_id'] | 0.1214 | 0.1249 | 144 | 164 | 149 | 57468 | 4 | 4 |
| support_champion+side | ['ally_utility_champion_id', 'side'] | 0.1212 | 0.1246 | 255 | 313 | 274 | 57468 | 9 | 10 |
| botlane_champions | ['ally_utility_champion_id', 'ally_bottom_champion_id'] | 0.1394 | 0.1239 | 2518 | 5846 | 3447 | 57468 | 371 | 386 |
| botlane_champions+side | ['ally_utility_champion_id', 'ally_bottom_champion_id', 'side'] | 0.1391 | 0.1132 | 3845 | 9284 | 5329 | 57468 | 601 | 636 |
| sup_vs_enemy_sup_champion | ['ally_utility_champion_id', 'enemy_utility_champion_id'] | 0.1316 | 0.1200 | 2720 | 6330 | 3642 | 57468 | 388 | 414 |
| support_riot_class | ['ally_utility_class'] | 0.0264 | 0.0218 | 7 | 7 | 7 | 57468 | 0 | 0 |
| botlane_riot_classes | ['ally_utility_class', 'ally_bottom_class'] | 0.0325 | 0.0301 | 47 | 49 | 46 | 57468 | 0 | 0 |
| all_10_riot_classes | ['ally_top_class', 'ally_jungle_class', 'ally_middle_class', 'ally_bottom_class', 'ally_utility_class', 'enemy_top_class', 'enemy_jungle_class', 'enemy_middle_class', 'enemy_bottom_class', 'enemy_utility_class'] | 0.0186 | -0.1824 | 7225 | 149794 | 42476 | 57468 | 24556 | 25285 |
| support_archetype | ['ally_utility_archetype'] | 0.0837 | 0.0852 | 29 | 29 | 29 | 57468 | 0 | 0 |
| support_archetype+side | ['ally_utility_archetype', 'side'] | 0.0823 | 0.0852 | 58 | 58 | 58 | 57468 | 0 | 0 |
| botlane_archetypes | ['ally_utility_archetype', 'ally_bottom_archetype'] | 0.0932 | 0.0946 | 388 | 564 | 432 | 57468 | 12 | 13 |
| botlane_archetypes+side | ['ally_utility_archetype', 'ally_bottom_archetype', 'side'] | 0.0931 | 0.0930 | 635 | 1007 | 728 | 57468 | 28 | 29 |
| sup_vs_enemy_sup_archetype | ['ally_utility_archetype', 'enemy_utility_archetype'] | 0.0901 | 0.0944 | 385 | 573 | 433 | 57468 | 15 | 22 |
| sup+jungle_archetypes | ['ally_utility_archetype', 'ally_jungle_archetype'] | 0.0818 | 0.0841 | 390 | 541 | 423 | 57468 | 14 | 17 |
| sup+jungle_archetypes+side | ['ally_utility_archetype', 'ally_jungle_archetype', 'side'] | 0.0819 | 0.0822 | 653 | 986 | 737 | 57468 | 30 | 35 |
| sup+jungle+top_archetypes | ['ally_utility_archetype', 'ally_jungle_archetype', 'ally_top_archetype'] | 0.0816 | 0.0651 | 2650 | 5913 | 3502 | 57468 | 340 | 376 |
| botlane_vs_enemy_bot_archetypes | ['ally_utility_archetype', 'ally_bottom_archetype', 'enemy_utility_archetype', 'enemy_bottom_archetype'] | 0.1006 | 0.0617 | 4800 | 18737 | 8425 | 57468 | 1738 | 1850 |
| ally_team_archetypes | ['ally_top_archetype', 'ally_jungle_archetype', 'ally_middle_archetype', 'ally_bottom_archetype', 'ally_utility_archetype'] | 0.0769 | -0.0787 | 10122 | 76587 | 27271 | 57468 | 9741 | 10226 |
| ally_team_archetypes+side | ['ally_top_archetype', 'ally_jungle_archetype', 'ally_middle_archetype', 'ally_bottom_archetype', 'ally_utility_archetype', 'side'] | 0.0711 | -0.1160 | 10534 | 97992 | 32950 | 57468 | 13225 | 13812 |
| all_10_archetypes | ['ally_top_archetype', 'ally_jungle_archetype', 'ally_middle_archetype', 'ally_bottom_archetype', 'ally_utility_archetype', 'enemy_top_archetype', 'enemy_jungle_archetype', 'enemy_middle_archetype', 'enemy_bottom_archetype', 'enemy_utility_archetype'] | nan | -0.0023 | 0 | 267611 | 57432 | 57468 | 57124 | 57158 |
| all_10_archetypes+side | ['ally_top_archetype', 'ally_jungle_archetype', 'ally_middle_archetype', 'ally_bottom_archetype', 'ally_utility_archetype', 'enemy_top_archetype', 'enemy_jungle_archetype', 'enemy_middle_archetype', 'enemy_bottom_archetype', 'enemy_utility_archetype', 'side'] | nan | -0.0017 | 0 | 267928 | 57452 | 57468 | 57302 | 57318 |
