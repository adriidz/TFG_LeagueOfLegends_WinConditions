param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$RawRoot = "data\raw\raw",
    [string]$Region = "europe",
    [int]$MaxMatches = 0,
    [switch]$SkipFrameState,
    [switch]$SkipDraftFeatures,
    [switch]$ShuffleMatchDirs,
    [int]$Seed = 42,
    [ValidateSet("single", "dataset")]
    [string]$FrameStateWriteMode = "dataset",
    [int]$FrameStateChunkMatches = 10000,
    [switch]$NoOverwriteFrameState
)

$ErrorActionPreference = "Stop"

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

$maxMatchArgs = @()
if ($MaxMatches -gt 0) {
    $maxMatchArgs = @("--max-matches", "$MaxMatches")
}

$shuffleArgs = @()
if ($ShuffleMatchDirs) {
    $shuffleArgs = @("--shuffle-match-dirs", "--seed", "$Seed")
}

if (-not $SkipFrameState) {
    $frameStateWriteArgs = @("--write-mode", $FrameStateWriteMode)
    if ($FrameStateWriteMode -eq "dataset") {
        $frameStateWriteArgs += @("--chunk-matches", "$FrameStateChunkMatches")
    }
    if (-not $NoOverwriteFrameState) {
        $frameStateWriteArgs += @("--overwrite-output")
    }
    $extractArgs = @(
        "ProgresoActual\src\02_data_processing\new_02a_extract_support_frame_state.py",
        "--raw-root", $RawRoot,
        "--region", $Region
    ) + $maxMatchArgs + $shuffleArgs + $frameStateWriteArgs
    Invoke-Step "Extract full support frame state" $extractArgs
}

if (-not $SkipDraftFeatures) {
    $draftArgs = @(
        "ProgresoActual\src\02_data_processing\build_draft_features.py",
        "--raw-root", $RawRoot,
        "--region", $Region
    ) + $maxMatchArgs + $shuffleArgs
    Invoke-Step "Build full draft features" $draftArgs
}

Write-Host ""
Write-Host "Full snapshot finished."
Write-Host "Frame state:    ProgresoActual\data\clean\frame_state\support_frame_state.parquet"
Write-Host "Draft features: ProgresoActual\data\clean\features\draft_features.parquet"
Write-Host "Frame mode:     $FrameStateWriteMode"
Write-Host "Next step:      .\ProgresoActual\run_support_pipeline.ps1 -SampleFrac 1 -SkipFrameState -SkipDraftFeatures"
