param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$ExperimentName = "support_oat_sample5_m12",
    [string]$SampleTag = "sample5",
    [double]$SampleFrac = 0.05,
    [string]$WindowTag = "m12",
    [string]$DraftPath = "",
    [string]$FrameStateDir = "ProgresoActual/data/clean/frame_state",
    [string]$FrameStateName = "support_frame_state",
    [double]$BaselineStartMinute = 5,
    [double]$BaselineMaxMinute = 12,
    [double]$FarAdcThreshold = 2500,
    [string]$BaselineWeights = "0.45,0.35,0.20",
    [string]$FeatureGroups = "standard",
    [int]$BatchSize = 256,
    [int]$Epochs = 60,
    [double]$Lr = 1e-3,
    [int]$Hidden1 = 256,
    [int]$Hidden2 = 128,
    [double]$Dropout = 0.2,
    [double]$WeightDecay = 1e-5,
    [int]$Patience = 10,
    [double]$ValSize = 0.2,
    [int]$Seed = 42,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"

if ($Smoke -and $ExperimentName -eq "support_oat_sample5_m12") {
    $ExperimentName = "${ExperimentName}_smoke"
}

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$StepArgs
    )
    Write-Host ""
    Write-Host "==== $Name ===="
    Write-Host ($StepArgs -join " ")
    & $Python @StepArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name"
    }
}

function Assert-Path {
    param([string]$Path, [string]$Hint)
    if (-not (Test-Path $Path)) {
        throw "No existe: $Path. $Hint"
    }
}

function Safe-Token {
    param([string]$Value)
    return ($Value -replace "\.", "p" -replace ",", "-" -replace "[^A-Za-z0-9_-]", "")
}

function Format-Invariant {
    param([object]$Value)
    if ($Value -is [double] -or $Value -is [float] -or $Value -is [decimal]) {
        return ([double]$Value).ToString("G", [System.Globalization.CultureInfo]::InvariantCulture)
    }
    return "$Value"
}

function New-RunConfig {
    param(
        [string]$Phase,
        [string]$ChangedParameter,
        [string]$ChangedValue,
        [double]$StartMinute,
        [double]$MaxMinute,
        [string]$Weights,
        [string]$FeatureGroups,
        [int]$BatchSize,
        [int]$Epochs,
        [double]$Lr,
        [int]$Hidden1,
        [int]$Hidden2,
        [double]$Dropout,
        [double]$WeightDecay,
        [int]$Patience,
        [double]$ValSize,
        [int]$Seed
    )
    $id = "{0}_{1}_{2}" -f (Safe-Token $Phase), (Safe-Token $ChangedParameter), (Safe-Token $ChangedValue)
    return [ordered]@{
        phase = $Phase
        experiment_id = $id
        changed_parameter = $ChangedParameter
        changed_value = $ChangedValue
        start_minute = $StartMinute
        max_minute = $MaxMinute
        far_adc_threshold = $FarAdcThreshold
        weight_triplet = $Weights
        feature_groups = $FeatureGroups
        batch_size = $BatchSize
        epochs = $Epochs
        lr = $Lr
        hidden1 = $Hidden1
        hidden2 = $Hidden2
        dropout = $Dropout
        weight_decay = $WeightDecay
        patience = $Patience
        val_size = $ValSize
        seed = $Seed
    }
}

$sampleSuffix = if ($SampleTag -and $SampleTag -ne "full") { "_$SampleTag" } else { "" }
$sampleArgs = @()
if ($SampleFrac -gt 0 -and $SampleFrac -lt 1) {
    $sampleArgs = @("--sample-frac", "$SampleFrac")
}
if ([string]::IsNullOrWhiteSpace($DraftPath)) {
    $DraftPath = "ProgresoActual/data/clean/features/draft_features$sampleSuffix.parquet"
}
$frameStatePath = "$FrameStateDir/$FrameStateName$sampleSuffix.parquet"

Assert-Path $frameStatePath "Ejecuta primero .\ProgresoActual\run_support_pipeline.ps1 o genera support_frame_state."
Assert-Path $DraftPath "Ejecuta primero .\ProgresoActual\run_support_pipeline.ps1 o genera draft_features."

$experimentRoot = "ProgresoActual/experiments/support_oat/$ExperimentName"
$scoreRoot = "ProgresoActual/data/clean/scores/oat_tuning/$ExperimentName"
$modelInputRoot = "ProgresoActual/data/training/oat_tuning/$ExperimentName"
$labelPlotRoot = "ProgresoActual/analysis/support_label_distribution/oat_tuning/$ExperimentName"
$gridRoot = "ProgresoActual/analysis/support_grid/oat_tuning/$ExperimentName"
$modelRoot = "ProgresoActual/models/oat_tuning/$ExperimentName"

New-Item -ItemType Directory -Force -Path $experimentRoot, $scoreRoot, $modelInputRoot, $labelPlotRoot, $gridRoot | Out-Null

$runs = New-Object System.Collections.Generic.List[object]

$weightGrid = @(
    "0.55,0.25,0.20",
    "0.45,0.35,0.20",
    "0.35,0.45,0.20",
    "0.45,0.25,0.30",
    "0.35,0.35,0.30"
)
$startGrid = @(4, 5, 6)
$maxGrid = @(10, 12, 14)
$hparamRuns = @(
    @{ name = "lr"; value = "5e-4"; lr = 5e-4 },
    @{ name = "dropout"; value = "0.1"; dropout = 0.1 },
    @{ name = "dropout"; value = "0.3"; dropout = 0.3 },
    @{ name = "hidden"; value = "512-256"; hidden1 = 512; hidden2 = 256 },
    @{ name = "weight_decay"; value = "1e-4"; weight_decay = 1e-4 },
    @{ name = "batch_size"; value = "512"; batch_size = 512 }
)

foreach ($weights in $weightGrid) {
    $runs.Add((New-RunConfig "label_weights" "weight_triplet" $weights $BaselineStartMinute $BaselineMaxMinute $weights $FeatureGroups $BatchSize $Epochs $Lr $Hidden1 $Hidden2 $Dropout $WeightDecay $Patience $ValSize $Seed))
}
foreach ($s in $startGrid) {
    foreach ($m in $maxGrid) {
        if ($s -lt $m) {
            $runs.Add((New-RunConfig "time_window" "start_max" "s${s}_m${m}" $s $m $BaselineWeights $FeatureGroups $BatchSize $Epochs $Lr $Hidden1 $Hidden2 $Dropout $WeightDecay $Patience $ValSize $Seed))
        }
    }
}
foreach ($h in $hparamRuns) {
    $runLr = if ($h.ContainsKey("lr")) { [double]$h.lr } else { $Lr }
    $runDropout = if ($h.ContainsKey("dropout")) { [double]$h.dropout } else { $Dropout }
    $runHidden1 = if ($h.ContainsKey("hidden1")) { [int]$h.hidden1 } else { $Hidden1 }
    $runHidden2 = if ($h.ContainsKey("hidden2")) { [int]$h.hidden2 } else { $Hidden2 }
    $runWeightDecay = if ($h.ContainsKey("weight_decay")) { [double]$h.weight_decay } else { $WeightDecay }
    $runBatchSize = if ($h.ContainsKey("batch_size")) { [int]$h.batch_size } else { $BatchSize }
    $runs.Add((New-RunConfig "train_hparams" $h.name $h.value $BaselineStartMinute $BaselineMaxMinute $BaselineWeights $FeatureGroups $runBatchSize $Epochs $runLr $runHidden1 $runHidden2 $runDropout $runWeightDecay $Patience $ValSize $Seed))
}

if ($Smoke) {
    $Epochs = 2
    $runs = @(
        (New-RunConfig "label_weights" "weight_triplet" $BaselineWeights $BaselineStartMinute $BaselineMaxMinute $BaselineWeights $FeatureGroups $BatchSize 2 $Lr $Hidden1 $Hidden2 $Dropout $WeightDecay $Patience $ValSize $Seed),
        (New-RunConfig "time_window" "start_max" "s4_m12" 4 12 $BaselineWeights $FeatureGroups $BatchSize 2 $Lr $Hidden1 $Hidden2 $Dropout $WeightDecay $Patience $ValSize $Seed),
        (New-RunConfig "train_hparams" "dropout" "0.3" $BaselineStartMinute $BaselineMaxMinute $BaselineWeights $FeatureGroups $BatchSize 2 $Lr $Hidden1 $Hidden2 0.3 $WeightDecay $Patience $ValSize $Seed)
    )
}

$manifestRows = New-Object System.Collections.Generic.List[object]

foreach ($run in $runs) {
    $experimentId = $run.experiment_id
    $scoreDir = "$scoreRoot/$experimentId"
    $gridDir = "$gridRoot/$experimentId"
    $scorePath = "$scoreDir/support_scores.parquet"
    $supportConfigPath = "$scoreDir/selected_support_score_config.json"
    $modelInputPath = "$modelInputRoot/$experimentId/model_input.parquet"
    $labelDir = "$labelPlotRoot/$experimentId"
    $trainOutdir = "$modelRoot/$experimentId"

    New-Item -ItemType Directory -Force -Path $scoreDir, $gridDir, (Split-Path $modelInputPath), $labelDir | Out-Null

    $gridArgs = @(
        "ProgresoActual/src/02_data_processing/new_02b_grid_support_scores.py",
        "--frame-state-dir", $FrameStateDir,
        "--frame-state-name", $FrameStateName,
        "--outdir", $gridDir,
        "--start-minutes", "$($run.start_minute)",
        "--max-minutes", "$($run.max_minute)",
        "--far-adc-thresholds", "$($run.far_adc_threshold)",
        "--weight-triplets", $run.weight_triplet,
        "--champion-summary",
        "--export-best", "coverage",
        "--export-support-scores-path", $scorePath,
        "--write-config-json"
    ) + $sampleArgs
    Invoke-Step "Score $experimentId" $gridArgs

    Invoke-Step "Build model input $experimentId" @(
        "ProgresoActual/src/02_data_processing/build_support_model_input.py",
        "--draft-path", $DraftPath,
        "--support-scores-path", $scorePath,
        "--out-path", $modelInputPath
    )

    Invoke-Step "Plot label distribution $experimentId" @(
        "ProgresoActual/scripts/plot_support_label_distribution.py",
        "--support-scores-path", $scorePath,
        "--outdir", $labelDir
    )

    $row = [pscustomobject]@{
        phase = $run.phase
        experiment_id = $experimentId
        changed_parameter = $run.changed_parameter
        changed_value = $run.changed_value
        start_minute = Format-Invariant $run.start_minute
        max_minute = Format-Invariant $run.max_minute
        far_adc_threshold = Format-Invariant $run.far_adc_threshold
        weight_triplet = $run.weight_triplet
        support_scores_path = $scorePath
        support_config_json = $supportConfigPath
        model_input_path = $modelInputPath
        label_distribution_dir = $labelDir
        train_outdir = $trainOutdir
        feature_groups = $run.feature_groups
        batch_size = $run.batch_size
        epochs = $run.epochs
        lr = Format-Invariant $run.lr
        hidden1 = $run.hidden1
        hidden2 = $run.hidden2
        dropout = Format-Invariant $run.dropout
        weight_decay = Format-Invariant $run.weight_decay
        patience = $run.patience
        val_size = Format-Invariant $run.val_size
        seed = $run.seed
        objective_metric = "val_mse"
    }
    $manifestRows.Add($row)
}

$manifestPath = "$experimentRoot/runs_manifest.csv"
$manifestRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $manifestPath

Write-Host ""
Write-Host "OAT tuning preparation finished."
Write-Host "Experiment: $ExperimentName"
Write-Host "Runs:       $($manifestRows.Count)"
Write-Host "Manifest:   $manifestPath"
Write-Host "Next step:  .\ProgresoActual\scripts\sync_support_oat_to_cluster.ps1 -ExperimentName $ExperimentName"
