param(
    [string]$ClusterUser = "adiaz",
    [string]$ClusterHost = "158.109.75.51",
    [int]$Port = 55022,
    [string]$ClusterProjectDir = "/fhome/adiaz/TFG_LeagueOfLegends_WinConditions",
    [string]$ExperimentName = "support_oat_sample5_m12",
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
        throw "No existe el artefacto local requerido: $Path"
    }
}

$remote = "$ClusterUser@$ClusterHost"
$experimentRoot = "ProgresoActual\experiments\support_oat\$ExperimentName"
$scoreRoot = "ProgresoActual\data\clean\scores\oat_tuning\$ExperimentName"
$modelInputRoot = "ProgresoActual\data\training\oat_tuning\$ExperimentName"
$labelRoot = "ProgresoActual\analysis\support_label_distribution\oat_tuning\$ExperimentName"

Assert-LocalPath "$experimentRoot\runs_manifest.csv"
Assert-LocalPath $scoreRoot
Assert-LocalPath $modelInputRoot
Assert-LocalPath $labelRoot

$remoteExperimentParent = "$ClusterProjectDir/ProgresoActual/experiments/support_oat"
$remoteScoreParent = "$ClusterProjectDir/ProgresoActual/data/clean/scores/oat_tuning"
$remoteModelInputParent = "$ClusterProjectDir/ProgresoActual/data/training/oat_tuning"
$remoteLabelParent = "$ClusterProjectDir/ProgresoActual/analysis/support_label_distribution/oat_tuning"

Invoke-External $Ssh @(
    "-p", "$Port",
    $remote,
    "mkdir -p '$remoteExperimentParent' '$remoteScoreParent' '$remoteModelInputParent' '$remoteLabelParent'"
)

Invoke-External $Scp @("-P", "$Port", "-r", $experimentRoot, "${remote}:${remoteExperimentParent}/")
Invoke-External $Scp @("-P", "$Port", "-r", $scoreRoot, "${remote}:${remoteScoreParent}/")
Invoke-External $Scp @("-P", "$Port", "-r", $modelInputRoot, "${remote}:${remoteModelInputParent}/")
Invoke-External $Scp @("-P", "$Port", "-r", $labelRoot, "${remote}:${remoteLabelParent}/")

Write-Host ""
Write-Host "OAT sync finished."
Write-Host "Manifest on cluster: $ClusterProjectDir/ProgresoActual/experiments/support_oat/$ExperimentName/runs_manifest.csv"
Write-Host "Next step on cluster:"
Write-Host "  sbatch --array=1-N ProgresoActual/scripts/train_support_oat_array.sh"
