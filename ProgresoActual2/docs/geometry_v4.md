# Geometry v4

`geometry_v4` is an isolated geometry redesign for the second progress phase.
It does not modify `ProgresoActual/src/02_data_processing/shared_utils.py`.

The goal is to derive a walkable map mask from observed all-player timeline
coordinates and then classify semantic zones on top of that mask.

## Build

Default build uses 50k shuffled matches and creates both windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\ProgresoActual2\scripts\run_geometry_v4_build.ps1
```

Use all available matches if time allows:

```powershell
powershell -ExecutionPolicy Bypass -File .\ProgresoActual2\scripts\run_geometry_v4_build.ps1 -MaxMatches 0
```

## Outputs

Data artifacts:

```text
ProgresoActual2/data/geometry/observed_player_density_0_14.npz
ProgresoActual2/data/geometry/observed_walkable_mask_0_14.npz
ProgresoActual2/data/geometry/observed_player_density_5_12.npz
ProgresoActual2/data/geometry/observed_walkable_mask_5_12.npz
```

Visual diagnostics:

```text
ProgresoActual2/analysis/geometry_v4/m0_14/
ProgresoActual2/analysis/geometry_v4/m5_12/
```

Key files:

- `geometry_v4_walkable_mask.png`
- `geometry_v4_walkable_mask_on_heatmap.png`
- `geometry_v2_outlines_on_heatmap_blue.png`
- `geometry_v4_outlines_on_heatmap_blue.png`
- `geometry_v4_zone_layer_blue.png`
- `support_zone_v2_to_v4_sample.csv`

## API

The module lives in:

```text
ProgresoActual2/src/geometry/geometry_v4.py
```

Main functions:

```python
classify_zone_v4(x, y, team_id)
is_walkable_v4(x, y)
distance_to_bot_lane_v4(x, y, team_id)
bot_distance_signal_v4(x, y, team_id)
is_in_bot_context_v4(x, y, team_id)
```

The default mask is `observed_walkable_mask_0_14.npz`. Pass `mask_path` to use
the `5-12` mask or another experimental mask.
