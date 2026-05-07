param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [ValidateSet("smoke", "full")]
    [string]$Mode = "smoke",
    [double]$SampleFrac = 0.05,
    [int]$Seed = 42,
    [switch]$ExportSelected
)

$ErrorActionPreference = "Stop"

$argsList = @(
    "ProgresoActual2\scripts\compare_support_label_variants.py",
    "--mode", $Mode,
    "--sample-frac", "$SampleFrac",
    "--seed", "$Seed"
)

if ($ExportSelected) {
    $argsList += "--export-selected"
}

Write-Host ($Python + " " + ($argsList -join " "))
& $Python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Support label variant comparison failed."
}
