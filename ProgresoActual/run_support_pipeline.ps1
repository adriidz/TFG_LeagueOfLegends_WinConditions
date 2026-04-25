param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$RawRoot = "data\raw\raw",
    [string]$Region = "europe",
    [double]$SampleFrac = 0.05,
    [int]$MaxMatches = 0,
    [double[]]$StartMinutes = @(5),
    [double[]]$MaxMinutes = @(11),
    [double[]]$FarAdcThresholds = @(2500),
    [string[]]$WeightTriplets = @("0.45,0.35,0.20"),
    [string]$ExportBest = "coverage",
    [int]$Epochs = 40,
    [int]$BatchSize = 512,
    [string]$FeatureGroups = "standard",
    [switch]$SkipFrameState,
    [switch]$SkipDraftFeatures,
    [switch]$SkipTraining,
    [switch]$SkipChampionAnalysis,
    [switch]$Wandb,
    [string]$WandbMode = "offline"
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Args
    )
    Write-Host ""
    Write-Host "==== $Name ===="
    Write-Host ($Args -join " ")
    & $Python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name"
    }
}

$sampleArgs = @()
$sampleSuffix = ""
if ($SampleFrac -gt 0 -and $SampleFrac -lt 1) {
    $sampleArgs = @("--sample-frac", "$SampleFrac")
    $sampleSuffix = "_sample$([int][Math]::Round($SampleFrac * 100))"
}

$maxMatchArgs = @()
if ($MaxMatches -gt 0) {
    $maxMatchArgs = @("--max-matches", "$MaxMatches")
}

$scoreMinute = $MaxMinutes[0]
$scoreMinuteInt = [int][Math]::Round($scoreMinute)
$windowTag = "m{0:D2}" -f $scoreMinuteInt

$frameStatePath = "ProgresoActual\data\clean\frame_state\support_frame_state$sampleSuffix.parquet"
$draftPath = "ProgresoActual\data\clean\features\draft_features$sampleSuffix.parquet"
$supportScoresPath = "ProgresoActual\data\clean\scores\support_scores$sampleSuffix`_$windowTag.parquet"
$supportConfigPath = "ProgresoActual\data\clean\scores\selected_support_score_config.json"
$modelInputPath = "ProgresoActual\data\training\model_input_support_regression$sampleSuffix`_$windowTag.parquet"
$modelOutDir = "ProgresoActual\models\support_mlp_regression$sampleSuffix`_$windowTag"

if (-not $SkipFrameState) {
    Invoke-Step "Extract support frame state" @(
        "ProgresoActual\src\02_data_processing\new_02a_extract_support_frame_state.py",
        "--raw-root", $RawRoot,
        "--region", $Region
    ) + $sampleArgs + $maxMatchArgs
}

if (-not $SkipDraftFeatures) {
    Invoke-Step "Build draft features" @(
        "ProgresoActual\src\02_data_processing\build_draft_features.py",
        "--raw-root", $RawRoot,
        "--region", $Region
    ) + $sampleArgs + $maxMatchArgs
}

$gridArgs = @(
    "ProgresoActual\src\02_data_processing\new_02b_grid_support_scores.py",
    "--frame-state-dir", "ProgresoActual\data\clean\frame_state",
    "--outdir", "ProgresoActual\analysis\support_grid",
    "--start-minutes"
) + ($StartMinutes | ForEach-Object { "$_" }) + @(
    "--max-minutes"
) + ($MaxMinutes | ForEach-Object { "$_" }) + @(
    "--far-adc-thresholds"
) + ($FarAdcThresholds | ForEach-Object { "$_" }) + @(
    "--weight-triplets"
) + $WeightTriplets + @(
    "--champion-summary",
    "--export-best", $ExportBest,
    "--export-support-scores-path", $supportScoresPath,
    "--write-config-json"
) + $sampleArgs
Invoke-Step "Grid/export support scores" $gridArgs

Invoke-Step "Build support model input" @(
    "ProgresoActual\src\02_data_processing\build_support_model_input.py",
    "--draft-path", $draftPath,
    "--support-scores-path", $supportScoresPath,
    "--out-path", $modelInputPath
)

if (-not $SkipTraining) {
    $trainArgs = @(
        "ProgresoActual\scripts\train_support_mlp_regression.py",
        "--input", $modelInputPath,
        "--outdir", $modelOutDir,
        "--feature-groups", $FeatureGroups,
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--support-config-json", $supportConfigPath
    )
    if ($Wandb) {
        $trainArgs += @("--wandb", "--wandb-mode", $WandbMode)
    }
    Invoke-Step "Train support MLP" $trainArgs
}

if (-not $SkipChampionAnalysis) {
    Invoke-Step "Build champion reference" @(
        "ProgresoActual\scripts\build_champion_support_reference.py",
        "--manual-only"
    )
    Invoke-Step "Compare champion reference" @(
        "ProgresoActual\scripts\compare_support_champion_reference.py",
        "--support-scores-path", $supportScoresPath,
        "--reference-path", "ProgresoActual\references\champion_support_reference.csv",
        "--outdir", "ProgresoActual\analysis\champion_reference"
    )
}

Write-Host ""
Write-Host "Pipeline finished."
Write-Host "Support scores: $supportScoresPath"
Write-Host "Model input:    $modelInputPath"
Write-Host "Model outdir:   $modelOutDir"
