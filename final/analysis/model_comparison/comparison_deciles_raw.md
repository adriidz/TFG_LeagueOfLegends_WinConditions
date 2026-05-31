# Raw-Scale Prediction Deciles

## Prediction Decile Diagnostics

| model                                       | trained_target | decile | n    | pred_mean | true_mean | true_std | abs_error_mean | true_q25 | true_q75 |
| ------------------------------------------- | -------------- | ------ | ---- | --------- | --------- | -------- | -------------- | -------- | -------- |
| Champion Mean                               | raw            | 1      | 5747 | 0.2880    | 0.2887    | 0.1810   | 0.1384         | 0.1467   | 0.4105   |
| Champion Mean                               | raw            | 2      | 5747 | 0.3248    | 0.3223    | 0.1796   | 0.1458         | 0.1838   | 0.4404   |
| Champion Mean                               | raw            | 3      | 5747 | 0.3440    | 0.3383    | 0.1808   | 0.1470         | 0.2054   | 0.4617   |
| Champion Mean                               | raw            | 4      | 5747 | 0.3574    | 0.3587    | 0.1807   | 0.1460         | 0.2262   | 0.4815   |
| Champion Mean                               | raw            | 5      | 5747 | 0.3825    | 0.3806    | 0.1766   | 0.1425         | 0.2541   | 0.5017   |
| Champion Mean                               | raw            | 6      | 5747 | 0.4048    | 0.4077    | 0.1756   | 0.1421         | 0.2833   | 0.5303   |
| Champion Mean                               | raw            | 7      | 5747 | 0.4172    | 0.4163    | 0.1767   | 0.1433         | 0.2935   | 0.5422   |
| Champion Mean                               | raw            | 8      | 5747 | 0.4323    | 0.4261    | 0.1817   | 0.1462         | 0.2991   | 0.5489   |
| Champion Mean                               | raw            | 9      | 5746 | 0.4612    | 0.4651    | 0.1817   | 0.1452         | 0.3414   | 0.5869   |
| Champion Mean                               | raw            | 10     | 5746 | 0.5101    | 0.5116    | 0.1791   | 0.1430         | 0.3915   | 0.6344   |
| Champion Mean (quantile->raw)               | quantile       | 1      | 5747 | 0.3050    | 0.2887    | 0.1809   | 0.1409         | 0.1467   | 0.4105   |
| Champion Mean (quantile->raw)               | quantile       | 2      | 5747 | 0.3357    | 0.3223    | 0.1797   | 0.1468         | 0.1837   | 0.4405   |
| Champion Mean (quantile->raw)               | quantile       | 3      | 5747 | 0.3514    | 0.3382    | 0.1807   | 0.1476         | 0.2054   | 0.4615   |
| Champion Mean (quantile->raw)               | quantile       | 4      | 5747 | 0.3624    | 0.3588    | 0.1808   | 0.1462         | 0.2262   | 0.4815   |
| Champion Mean (quantile->raw)               | quantile       | 5      | 5747 | 0.3815    | 0.3806    | 0.1765   | 0.1425         | 0.2542   | 0.5020   |
| Champion Mean (quantile->raw)               | quantile       | 6      | 5747 | 0.3989    | 0.4077    | 0.1757   | 0.1423         | 0.2833   | 0.5304   |
| Champion Mean (quantile->raw)               | quantile       | 7      | 5747 | 0.4097    | 0.4163    | 0.1774   | 0.1440         | 0.2929   | 0.5425   |
| Champion Mean (quantile->raw)               | quantile       | 8      | 5747 | 0.4202    | 0.4259    | 0.1810   | 0.1456         | 0.2991   | 0.5478   |
| Champion Mean (quantile->raw)               | quantile       | 9      | 5746 | 0.4430    | 0.4654    | 0.1816   | 0.1464         | 0.3414   | 0.5877   |
| Champion Mean (quantile->raw)               | quantile       | 10     | 5746 | 0.4808    | 0.5116    | 0.1791   | 0.1460         | 0.3915   | 0.6344   |
| Global Mean                                 | raw            | 1      | 5747 | 0.3916    | 0.3929    | 0.1893   | 0.1536         | 0.2580   | 0.5235   |
| Global Mean                                 | raw            | 2      | 5747 | 0.3916    | 0.3899    | 0.1908   | 0.1555         | 0.2492   | 0.5227   |
| Global Mean                                 | raw            | 3      | 5747 | 0.3916    | 0.3952    | 0.1871   | 0.1520         | 0.2627   | 0.5267   |
| Global Mean                                 | raw            | 4      | 5747 | 0.3916    | 0.3944    | 0.1912   | 0.1564         | 0.2521   | 0.5334   |
| Global Mean                                 | raw            | 5      | 5747 | 0.3916    | 0.3880    | 0.1920   | 0.1573         | 0.2484   | 0.5251   |
| Global Mean                                 | raw            | 6      | 5747 | 0.3916    | 0.3877    | 0.1901   | 0.1541         | 0.2533   | 0.5157   |
| Global Mean                                 | raw            | 7      | 5747 | 0.3916    | 0.3963    | 0.1904   | 0.1557         | 0.2540   | 0.5310   |
| Global Mean                                 | raw            | 8      | 5747 | 0.3916    | 0.3973    | 0.1901   | 0.1546         | 0.2585   | 0.5296   |
| Global Mean                                 | raw            | 9      | 5746 | 0.3916    | 0.3867    | 0.1913   | 0.1555         | 0.2446   | 0.5176   |
| Global Mean                                 | raw            | 10     | 5746 | 0.3916    | 0.3871    | 0.1926   | 0.1567         | 0.2488   | 0.5195   |
| Global Mean (quantile->raw)                 | quantile       | 1      | 5747 | 0.3884    | 0.3929    | 0.1893   | 0.1536         | 0.2580   | 0.5235   |
| Global Mean (quantile->raw)                 | quantile       | 2      | 5747 | 0.3884    | 0.3899    | 0.1908   | 0.1555         | 0.2492   | 0.5227   |
| Global Mean (quantile->raw)                 | quantile       | 3      | 5747 | 0.3884    | 0.3952    | 0.1871   | 0.1520         | 0.2627   | 0.5267   |
| Global Mean (quantile->raw)                 | quantile       | 4      | 5747 | 0.3884    | 0.3944    | 0.1912   | 0.1564         | 0.2521   | 0.5334   |
| Global Mean (quantile->raw)                 | quantile       | 5      | 5747 | 0.3884    | 0.3880    | 0.1920   | 0.1572         | 0.2484   | 0.5251   |
| Global Mean (quantile->raw)                 | quantile       | 6      | 5747 | 0.3884    | 0.3877    | 0.1901   | 0.1540         | 0.2533   | 0.5157   |
| Global Mean (quantile->raw)                 | quantile       | 7      | 5747 | 0.3884    | 0.3963    | 0.1904   | 0.1558         | 0.2540   | 0.5310   |
| Global Mean (quantile->raw)                 | quantile       | 8      | 5747 | 0.3884    | 0.3973    | 0.1901   | 0.1547         | 0.2585   | 0.5296   |
| Global Mean (quantile->raw)                 | quantile       | 9      | 5746 | 0.3884    | 0.3867    | 0.1913   | 0.1554         | 0.2446   | 0.5176   |
| Global Mean (quantile->raw)                 | quantile       | 10     | 5746 | 0.3884    | 0.3871    | 0.1926   | 0.1566         | 0.2488   | 0.5195   |
| HistGBT                                     | raw            | 1      | 5747 | 0.2648    | 0.2638    | 0.1719   | 0.1323         | 0.1344   | 0.3747   |
| HistGBT                                     | raw            | 2      | 5747 | 0.3118    | 0.3165    | 0.1745   | 0.1410         | 0.1883   | 0.4330   |
| HistGBT                                     | raw            | 3      | 5747 | 0.3339    | 0.3370    | 0.1747   | 0.1404         | 0.2075   | 0.4517   |
| HistGBT                                     | raw            | 4      | 5747 | 0.3544    | 0.3582    | 0.1755   | 0.1417         | 0.2284   | 0.4778   |
| HistGBT                                     | raw            | 5      | 5747 | 0.3742    | 0.3790    | 0.1756   | 0.1425         | 0.2520   | 0.5003   |
| HistGBT                                     | raw            | 6      | 5747 | 0.3931    | 0.3988    | 0.1780   | 0.1434         | 0.2744   | 0.5191   |
| HistGBT                                     | raw            | 7      | 5747 | 0.4130    | 0.4172    | 0.1770   | 0.1428         | 0.2924   | 0.5377   |
| HistGBT                                     | raw            | 8      | 5747 | 0.4374    | 0.4465    | 0.1742   | 0.1411         | 0.3248   | 0.5657   |
| HistGBT                                     | raw            | 9      | 5746 | 0.4685    | 0.4681    | 0.1764   | 0.1418         | 0.3501   | 0.5884   |
| HistGBT                                     | raw            | 10     | 5746 | 0.5208    | 0.5303    | 0.1771   | 0.1412         | 0.4122   | 0.6530   |
| HistGBT (quantile->raw)                     | quantile       | 1      | 5747 | 0.2851    | 0.2645    | 0.1730   | 0.1369         | 0.1334   | 0.3768   |
| HistGBT (quantile->raw)                     | quantile       | 2      | 5747 | 0.3249    | 0.3148    | 0.1721   | 0.1391         | 0.1879   | 0.4272   |
| HistGBT (quantile->raw)                     | quantile       | 3      | 5747 | 0.3437    | 0.3379    | 0.1758   | 0.1426         | 0.2072   | 0.4564   |
| HistGBT (quantile->raw)                     | quantile       | 4      | 5747 | 0.3603    | 0.3587    | 0.1773   | 0.1429         | 0.2291   | 0.4792   |
| HistGBT (quantile->raw)                     | quantile       | 5      | 5747 | 0.3753    | 0.3788    | 0.1746   | 0.1413         | 0.2530   | 0.4973   |
| HistGBT (quantile->raw)                     | quantile       | 6      | 5747 | 0.3901    | 0.3984    | 0.1775   | 0.1435         | 0.2734   | 0.5192   |
| HistGBT (quantile->raw)                     | quantile       | 7      | 5747 | 0.4061    | 0.4185    | 0.1775   | 0.1436         | 0.2938   | 0.5396   |
| HistGBT (quantile->raw)                     | quantile       | 8      | 5747 | 0.4258    | 0.4442    | 0.1740   | 0.1411         | 0.3234   | 0.5646   |
| HistGBT (quantile->raw)                     | quantile       | 9      | 5746 | 0.4500    | 0.4712    | 0.1767   | 0.1435         | 0.3506   | 0.5918   |
| HistGBT (quantile->raw)                     | quantile       | 10     | 5746 | 0.4896    | 0.5285    | 0.1770   | 0.1446         | 0.4106   | 0.6517   |
| HistGBT + Archetypes                        | raw            | 1      | 5747 | 0.2736    | 0.2627    | 0.1714   | 0.1335         | 0.1337   | 0.3742   |
| HistGBT + Archetypes                        | raw            | 2      | 5747 | 0.3190    | 0.3147    | 0.1725   | 0.1393         | 0.1874   | 0.4274   |
| HistGBT + Archetypes                        | raw            | 3      | 5747 | 0.3408    | 0.3388    | 0.1759   | 0.1425         | 0.2072   | 0.4551   |
| HistGBT + Archetypes                        | raw            | 4      | 5747 | 0.3608    | 0.3580    | 0.1776   | 0.1428         | 0.2293   | 0.4776   |
| HistGBT + Archetypes                        | raw            | 5      | 5747 | 0.3798    | 0.3818    | 0.1750   | 0.1421         | 0.2549   | 0.5034   |
| HistGBT + Archetypes                        | raw            | 6      | 5747 | 0.3980    | 0.3982    | 0.1767   | 0.1428         | 0.2736   | 0.5221   |
| HistGBT + Archetypes                        | raw            | 7      | 5747 | 0.4176    | 0.4172    | 0.1769   | 0.1428         | 0.2905   | 0.5379   |
| HistGBT + Archetypes                        | raw            | 8      | 5747 | 0.4417    | 0.4438    | 0.1756   | 0.1415         | 0.3236   | 0.5654   |
| HistGBT + Archetypes                        | raw            | 9      | 5746 | 0.4727    | 0.4713    | 0.1752   | 0.1405         | 0.3538   | 0.5893   |
| HistGBT + Archetypes                        | raw            | 10     | 5746 | 0.5229    | 0.5288    | 0.1776   | 0.1410         | 0.4093   | 0.6526   |
| HistGBT + Archetypes (quantile->raw)        | quantile       | 1      | 5747 | 0.2911    | 0.2629    | 0.1723   | 0.1379         | 0.1332   | 0.3749   |
| HistGBT + Archetypes (quantile->raw)        | quantile       | 2      | 5747 | 0.3302    | 0.3151    | 0.1715   | 0.1398         | 0.1887   | 0.4282   |
| HistGBT + Archetypes (quantile->raw)        | quantile       | 3      | 5747 | 0.3487    | 0.3398    | 0.1756   | 0.1419         | 0.2086   | 0.4534   |
| HistGBT + Archetypes (quantile->raw)        | quantile       | 4      | 5747 | 0.3649    | 0.3583    | 0.1771   | 0.1430         | 0.2291   | 0.4793   |
| HistGBT + Archetypes (quantile->raw)        | quantile       | 5      | 5747 | 0.3796    | 0.3796    | 0.1777   | 0.1443         | 0.2515   | 0.5029   |
| HistGBT + Archetypes (quantile->raw)        | quantile       | 6      | 5747 | 0.3941    | 0.3988    | 0.1757   | 0.1420         | 0.2745   | 0.5182   |
| HistGBT + Archetypes (quantile->raw)        | quantile       | 7      | 5747 | 0.4101    | 0.4162    | 0.1771   | 0.1432         | 0.2925   | 0.5379   |
| HistGBT + Archetypes (quantile->raw)        | quantile       | 8      | 5747 | 0.4291    | 0.4454    | 0.1745   | 0.1413         | 0.3251   | 0.5663   |
| HistGBT + Archetypes (quantile->raw)        | quantile       | 9      | 5746 | 0.4534    | 0.4698    | 0.1767   | 0.1429         | 0.3513   | 0.5908   |
| HistGBT + Archetypes (quantile->raw)        | quantile       | 10     | 5746 | 0.4919    | 0.5296    | 0.1765   | 0.1443         | 0.4129   | 0.6526   |
| HistGBT + Pair TE                           | raw            | 1      | 5747 | 0.2714    | 0.2635    | 0.1713   | 0.1327         | 0.1337   | 0.3750   |
| HistGBT + Pair TE                           | raw            | 2      | 5747 | 0.3174    | 0.3155    | 0.1727   | 0.1395         | 0.1869   | 0.4304   |
| HistGBT + Pair TE                           | raw            | 3      | 5747 | 0.3398    | 0.3381    | 0.1750   | 0.1412         | 0.2081   | 0.4531   |
| HistGBT + Pair TE                           | raw            | 4      | 5747 | 0.3602    | 0.3592    | 0.1777   | 0.1433         | 0.2291   | 0.4801   |
| HistGBT + Pair TE                           | raw            | 5      | 5747 | 0.3801    | 0.3778    | 0.1763   | 0.1430         | 0.2517   | 0.4977   |
| HistGBT + Pair TE                           | raw            | 6      | 5747 | 0.3993    | 0.3984    | 0.1764   | 0.1428         | 0.2713   | 0.5205   |
| HistGBT + Pair TE                           | raw            | 7      | 5747 | 0.4191    | 0.4179    | 0.1778   | 0.1430         | 0.2954   | 0.5385   |
| HistGBT + Pair TE                           | raw            | 8      | 5747 | 0.4432    | 0.4464    | 0.1743   | 0.1406         | 0.3260   | 0.5654   |
| HistGBT + Pair TE                           | raw            | 9      | 5746 | 0.4733    | 0.4689    | 0.1763   | 0.1415         | 0.3511   | 0.5878   |
| HistGBT + Pair TE                           | raw            | 10     | 5746 | 0.5230    | 0.5298    | 0.1769   | 0.1405         | 0.4118   | 0.6519   |
| HistGBT + Pair TE (quantile->raw)           | quantile       | 1      | 5747 | 0.2906    | 0.2620    | 0.1702   | 0.1358         | 0.1337   | 0.3723   |
| HistGBT + Pair TE (quantile->raw)           | quantile       | 2      | 5747 | 0.3299    | 0.3154    | 0.1729   | 0.1413         | 0.1860   | 0.4307   |
| HistGBT + Pair TE (quantile->raw)           | quantile       | 3      | 5747 | 0.3483    | 0.3370    | 0.1745   | 0.1416         | 0.2080   | 0.4503   |
| HistGBT + Pair TE (quantile->raw)           | quantile       | 4      | 5747 | 0.3647    | 0.3595    | 0.1776   | 0.1431         | 0.2297   | 0.4802   |
| HistGBT + Pair TE (quantile->raw)           | quantile       | 5      | 5747 | 0.3799    | 0.3827    | 0.1760   | 0.1423         | 0.2581   | 0.5030   |
| HistGBT + Pair TE (quantile->raw)           | quantile       | 6      | 5747 | 0.3948    | 0.3965    | 0.1777   | 0.1440         | 0.2699   | 0.5213   |
| HistGBT + Pair TE (quantile->raw)           | quantile       | 7      | 5747 | 0.4108    | 0.4189    | 0.1778   | 0.1437         | 0.2945   | 0.5400   |
| HistGBT + Pair TE (quantile->raw)           | quantile       | 8      | 5747 | 0.4294    | 0.4450    | 0.1748   | 0.1415         | 0.3236   | 0.5646   |
| HistGBT + Pair TE (quantile->raw)           | quantile       | 9      | 5746 | 0.4532    | 0.4688    | 0.1764   | 0.1424         | 0.3499   | 0.5879   |
| HistGBT + Pair TE (quantile->raw)           | quantile       | 10     | 5746 | 0.4912    | 0.5297    | 0.1761   | 0.1440         | 0.4117   | 0.6524   |
| MLP Embed                                   | raw            | 1      | 5747 | 0.2655    | 0.2684    | 0.1734   | 0.1326         | 0.1361   | 0.3792   |
| MLP Embed                                   | raw            | 2      | 5747 | 0.3136    | 0.3166    | 0.1737   | 0.1408         | 0.1878   | 0.4330   |
| MLP Embed                                   | raw            | 3      | 5747 | 0.3361    | 0.3394    | 0.1775   | 0.1433         | 0.2068   | 0.4581   |
| MLP Embed                                   | raw            | 4      | 5747 | 0.3564    | 0.3635    | 0.1792   | 0.1447         | 0.2318   | 0.4861   |
| MLP Embed                                   | raw            | 5      | 5747 | 0.3757    | 0.3775    | 0.1754   | 0.1410         | 0.2534   | 0.4951   |
| MLP Embed                                   | raw            | 6      | 5747 | 0.3942    | 0.3974    | 0.1767   | 0.1433         | 0.2695   | 0.5186   |
| MLP Embed                                   | raw            | 7      | 5747 | 0.4136    | 0.4161    | 0.1775   | 0.1433         | 0.2951   | 0.5396   |
| MLP Embed                                   | raw            | 8      | 5747 | 0.4360    | 0.4404    | 0.1777   | 0.1431         | 0.3157   | 0.5627   |
| MLP Embed                                   | raw            | 9      | 5746 | 0.4644    | 0.4734    | 0.1770   | 0.1424         | 0.3560   | 0.5940   |
| MLP Embed                                   | raw            | 10     | 5746 | 0.5154    | 0.5227    | 0.1771   | 0.1411         | 0.4070   | 0.6464   |
| MLP Embed (quantile->raw)                   | quantile       | 1      | 5747 | 0.2822    | 0.2686    | 0.1735   | 0.1355         | 0.1361   | 0.3829   |
| MLP Embed (quantile->raw)                   | quantile       | 2      | 5747 | 0.3245    | 0.3161    | 0.1730   | 0.1400         | 0.1883   | 0.4295   |
| MLP Embed (quantile->raw)                   | quantile       | 3      | 5747 | 0.3436    | 0.3400    | 0.1775   | 0.1435         | 0.2094   | 0.4583   |
| MLP Embed (quantile->raw)                   | quantile       | 4      | 5747 | 0.3611    | 0.3579    | 0.1777   | 0.1432         | 0.2276   | 0.4787   |
| MLP Embed (quantile->raw)                   | quantile       | 5      | 5747 | 0.3775    | 0.3804    | 0.1772   | 0.1432         | 0.2543   | 0.4976   |
| MLP Embed (quantile->raw)                   | quantile       | 6      | 5747 | 0.3937    | 0.3987    | 0.1773   | 0.1428         | 0.2766   | 0.5183   |
| MLP Embed (quantile->raw)                   | quantile       | 7      | 5747 | 0.4108    | 0.4174    | 0.1779   | 0.1437         | 0.2938   | 0.5407   |
| MLP Embed (quantile->raw)                   | quantile       | 8      | 5747 | 0.4301    | 0.4417    | 0.1778   | 0.1435         | 0.3204   | 0.5637   |
| MLP Embed (quantile->raw)                   | quantile       | 9      | 5746 | 0.4544    | 0.4716    | 0.1752   | 0.1425         | 0.3512   | 0.5918   |
| MLP Embed (quantile->raw)                   | quantile       | 10     | 5746 | 0.4934    | 0.5230    | 0.1775   | 0.1435         | 0.4030   | 0.6474   |
| MLP OneHot                                  | raw            | 1      | 5747 | 0.2597    | 0.2689    | 0.1731   | 0.1320         | 0.1361   | 0.3831   |
| MLP OneHot                                  | raw            | 2      | 5747 | 0.3103    | 0.3138    | 0.1726   | 0.1401         | 0.1827   | 0.4290   |
| MLP OneHot                                  | raw            | 3      | 5747 | 0.3332    | 0.3384    | 0.1776   | 0.1428         | 0.2076   | 0.4536   |
| MLP OneHot                                  | raw            | 4      | 5747 | 0.3540    | 0.3556    | 0.1771   | 0.1435         | 0.2237   | 0.4780   |
| MLP OneHot                                  | raw            | 5      | 5747 | 0.3746    | 0.3846    | 0.1760   | 0.1420         | 0.2551   | 0.5029   |
| MLP OneHot                                  | raw            | 6      | 5747 | 0.3949    | 0.3984    | 0.1774   | 0.1443         | 0.2714   | 0.5222   |
| MLP OneHot                                  | raw            | 7      | 5747 | 0.4162    | 0.4171    | 0.1768   | 0.1424         | 0.2952   | 0.5396   |
| MLP OneHot                                  | raw            | 8      | 5747 | 0.4418    | 0.4403    | 0.1762   | 0.1419         | 0.3164   | 0.5623   |
| MLP OneHot                                  | raw            | 9      | 5746 | 0.4760    | 0.4718    | 0.1783   | 0.1436         | 0.3510   | 0.5938   |
| MLP OneHot                                  | raw            | 10     | 5746 | 0.5357    | 0.5266    | 0.1755   | 0.1395         | 0.4082   | 0.6490   |
| MLP OneHot (quantile->raw)                  | quantile       | 1      | 5747 | 0.2786    | 0.2684    | 0.1720   | 0.1343         | 0.1370   | 0.3805   |
| MLP OneHot (quantile->raw)                  | quantile       | 2      | 5747 | 0.3206    | 0.3168    | 0.1754   | 0.1422         | 0.1852   | 0.4344   |
| MLP OneHot (quantile->raw)                  | quantile       | 3      | 5747 | 0.3393    | 0.3367    | 0.1767   | 0.1428         | 0.2059   | 0.4546   |
| MLP OneHot (quantile->raw)                  | quantile       | 4      | 5747 | 0.3558    | 0.3597    | 0.1770   | 0.1425         | 0.2316   | 0.4794   |
| MLP OneHot (quantile->raw)                  | quantile       | 5      | 5747 | 0.3707    | 0.3802    | 0.1771   | 0.1433         | 0.2520   | 0.5016   |
| MLP OneHot (quantile->raw)                  | quantile       | 6      | 5747 | 0.3851    | 0.3954    | 0.1773   | 0.1439         | 0.2665   | 0.5171   |
| MLP OneHot (quantile->raw)                  | quantile       | 7      | 5747 | 0.4000    | 0.4222    | 0.1783   | 0.1446         | 0.2991   | 0.5427   |
| MLP OneHot (quantile->raw)                  | quantile       | 8      | 5747 | 0.4174    | 0.4399    | 0.1738   | 0.1414         | 0.3211   | 0.5625   |
| MLP OneHot (quantile->raw)                  | quantile       | 9      | 5746 | 0.4391    | 0.4707    | 0.1773   | 0.1454         | 0.3514   | 0.5927   |
| MLP OneHot (quantile->raw)                  | quantile       | 10     | 5746 | 0.4789    | 0.5254    | 0.1774   | 0.1467         | 0.4058   | 0.6490   |
| MLP Per-Role + Interactions                 | raw            | 1      | 5747 | 0.2687    | 0.2674    | 0.1740   | 0.1343         | 0.1347   | 0.3798   |
| MLP Per-Role + Interactions                 | raw            | 2      | 5747 | 0.3131    | 0.3167    | 0.1715   | 0.1387         | 0.1884   | 0.4318   |
| MLP Per-Role + Interactions                 | raw            | 3      | 5747 | 0.3337    | 0.3389    | 0.1786   | 0.1427         | 0.2090   | 0.4523   |
| MLP Per-Role + Interactions                 | raw            | 4      | 5747 | 0.3540    | 0.3593    | 0.1763   | 0.1431         | 0.2289   | 0.4813   |
| MLP Per-Role + Interactions                 | raw            | 5      | 5747 | 0.3750    | 0.3778    | 0.1786   | 0.1440         | 0.2512   | 0.4967   |
| MLP Per-Role + Interactions                 | raw            | 6      | 5747 | 0.3957    | 0.3983    | 0.1761   | 0.1421         | 0.2735   | 0.5193   |
| MLP Per-Role + Interactions                 | raw            | 7      | 5747 | 0.4174    | 0.4172    | 0.1784   | 0.1446         | 0.2920   | 0.5416   |
| MLP Per-Role + Interactions                 | raw            | 8      | 5747 | 0.4425    | 0.4399    | 0.1757   | 0.1415         | 0.3177   | 0.5625   |
| MLP Per-Role + Interactions                 | raw            | 9      | 5746 | 0.4748    | 0.4737    | 0.1734   | 0.1391         | 0.3569   | 0.5917   |
| MLP Per-Role + Interactions                 | raw            | 10     | 5746 | 0.5304    | 0.5262    | 0.1782   | 0.1417         | 0.4072   | 0.6517   |
| MLP Per-Role + Interactions (quantile->raw) | quantile       | 1      | 5747 | 0.2868    | 0.2668    | 0.1723   | 0.1354         | 0.1361   | 0.3796   |
| MLP Per-Role + Interactions (quantile->raw) | quantile       | 2      | 5747 | 0.3252    | 0.3160    | 0.1750   | 0.1419         | 0.1841   | 0.4318   |
| MLP Per-Role + Interactions (quantile->raw) | quantile       | 3      | 5747 | 0.3421    | 0.3388    | 0.1752   | 0.1419         | 0.2088   | 0.4550   |
| MLP Per-Role + Interactions (quantile->raw) | quantile       | 4      | 5747 | 0.3585    | 0.3573    | 0.1782   | 0.1439         | 0.2271   | 0.4782   |
| MLP Per-Role + Interactions (quantile->raw) | quantile       | 5      | 5747 | 0.3750    | 0.3805    | 0.1751   | 0.1412         | 0.2564   | 0.4994   |
| MLP Per-Role + Interactions (quantile->raw) | quantile       | 6      | 5747 | 0.3917    | 0.4007    | 0.1775   | 0.1438         | 0.2742   | 0.5232   |
| MLP Per-Role + Interactions (quantile->raw) | quantile       | 7      | 5747 | 0.4100    | 0.4209    | 0.1791   | 0.1451         | 0.2958   | 0.5451   |
| MLP Per-Role + Interactions (quantile->raw) | quantile       | 8      | 5747 | 0.4299    | 0.4399    | 0.1780   | 0.1439         | 0.3168   | 0.5628   |
| MLP Per-Role + Interactions (quantile->raw) | quantile       | 9      | 5746 | 0.4558    | 0.4679    | 0.1751   | 0.1417         | 0.3489   | 0.5888   |
| MLP Per-Role + Interactions (quantile->raw) | quantile       | 10     | 5746 | 0.4986    | 0.5266    | 0.1761   | 0.1419         | 0.4093   | 0.6474   |
