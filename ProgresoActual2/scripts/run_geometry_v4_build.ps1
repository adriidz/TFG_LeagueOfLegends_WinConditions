param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [int]$MaxMatches = 50000,
    [int]$Workers = 1
)

$ErrorActionPreference = "Stop"

function Invoke-GeometryBuild {
    param([double]$StartMinute, [double]$MaxMinute)
    $argsList = @(
        "ProgresoActual2\scripts\build_geometry_v4_artifacts.py",
        "--start-minute", "$StartMinute",
        "--max-minute", "$MaxMinute",
        "--max-matches", "$MaxMatches",
        "--workers", "$Workers"
    )
    Write-Host ($Python + " " + ($argsList -join " "))
    & $Python @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "Geometry v4 build failed for $StartMinute-$MaxMinute."
    }
}

Invoke-GeometryBuild -StartMinute 0 -MaxMinute 14
Invoke-GeometryBuild -StartMinute 5 -MaxMinute 12
