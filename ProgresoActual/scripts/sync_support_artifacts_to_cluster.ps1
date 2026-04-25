param(
    [string]$ClusterUser = "adiaz",
    [string]$ClusterHost = "158.109.75.51",
    [int]$Port = 55022,
    [string]$ClusterProjectDir = "/fhome/adiaz/TFG_LeagueOfLegends_WinConditions",
    [string]$SampleTag = "sample5",
    [string]$WindowTag = "m12",
    [string]$Scp = "scp",
    [string]$Ssh = "ssh"
)

$ErrorActionPreference = "Stop"

function Invoke-External {
    param(
        [string]$Exe,
        [string[]]$CommandArgs
    )
    Write-Host ""
    Write-Host "==== $Exe ===="
    Write-Host ($CommandArgs -join " ")
    & $Exe @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Exe"
    }
}

function Assert-LocalPath {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        throw "No existe el artefacto local requerido: $Path. Ejecuta primero .\ProgresoActual\run_support_pipeline.ps1"
    }
}

$remote = "$ClusterUser@$ClusterHost"
$sampleSuffix = ""
if (-not [string]::IsNullOrWhiteSpace($SampleTag) -and $SampleTag -ne "full") {
    $sampleSuffix = "_$SampleTag"
}
$labelTag = if ([string]::IsNullOrWhiteSpace($SampleTag) -or $SampleTag -eq "full") {
    "full_$WindowTag"
} else {
    "${SampleTag}_${WindowTag}"
}

$modelInput = "ProgresoActual\data\training\model_input_support_regression$sampleSuffix`_$WindowTag.parquet"
$supportScores = "ProgresoActual\data\clean\scores\support_scores$sampleSuffix`_$WindowTag.parquet"
$supportConfig = "ProgresoActual\data\clean\scores\selected_support_score_config.json"
$labelDistribution = "ProgresoActual\analysis\support_label_distribution\$labelTag"

Assert-LocalPath $modelInput
Assert-LocalPath $supportScores
Assert-LocalPath $supportConfig
Assert-LocalPath $labelDistribution

$remoteTrainingDir = "$ClusterProjectDir/ProgresoActual/data/training"
$remoteScoresDir = "$ClusterProjectDir/ProgresoActual/data/clean/scores"
$remoteLabelParentDir = "$ClusterProjectDir/ProgresoActual/analysis/support_label_distribution"

Invoke-External $Ssh @(
    "-p", "$Port",
    $remote,
    "mkdir -p '$remoteTrainingDir' '$remoteScoresDir' '$remoteLabelParentDir'"
)

Invoke-External $Scp @(
    "-P", "$Port",
    $modelInput,
    "${remote}:${remoteTrainingDir}/"
)

Invoke-External $Scp @(
    "-P", "$Port",
    $supportScores,
    $supportConfig,
    "${remote}:${remoteScoresDir}/"
)

Invoke-External $Scp @(
    "-P", "$Port",
    "-r",
    $labelDistribution,
    "${remote}:${remoteLabelParentDir}/"
)

Write-Host ""
Write-Host "Sync finished."
Write-Host "Model input:    ${remote}:${remoteTrainingDir}/"
Write-Host "Support scores: ${remote}:${remoteScoresDir}/"
Write-Host "Label plots:    ${remote}:${remoteLabelParentDir}/$labelTag"
Write-Host "Next step on cluster: sbatch ProgresoActual/scripts/train_cluster_support_mlp.sh"
