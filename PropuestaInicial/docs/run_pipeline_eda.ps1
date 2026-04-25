$ErrorActionPreference = "Stop"

# ── CONFIGURACIÓN GENERAL ─────────────────────────────────────────────────────
$SampleFrac = "0.07"    # Muestreo al 5% para correlacionar con tus _sample5 existentes
$OutDir = "progreso_I"  # Toda la salida de estos scripts se guardará aquí

function Invoke-Step {
    param(
        [string]$Title,
        [string]$Command,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host $Title -ForegroundColor Cyan
    Write-Host ("Comando: " + $Command + " " + ($Arguments -join " ")) -ForegroundColor DarkGray
    Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host "Ejecutando (esto puede tardar varios minutos)..." -ForegroundColor Yellow

    & $Command @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw ("El comando fallo con exit code " + $LASTEXITCODE + "`n" + $Command + " " + ($Arguments -join " "))
    }

    Write-Host ("OK - " + $Title) -ForegroundColor Green
}

try {
    Write-Host "=== INICIO RUN PIPELINE EDA Y REPORTES ===" -ForegroundColor Yellow
    Write-Host ("Directorio Output    : " + $OutDir)
    Write-Host ("Sample Frac (Testeo) : " + $SampleFrac)
    Write-Host ""

    # Creamos outdir base por si no existe
    if (!(Test-Path $OutDir)) {
        New-Item -ItemType Directory -Path $OutDir | Out-Null
    }

    Invoke-Step `
        -Title "1/5 - Control de Calidad (QC Sanity Checks)" `
        -Command "python" `
        -Arguments @(
            "src/eda/qc_sanity_checks.py",
            "--out-dir", (Join-Path $OutDir "qc_reports"),
            "--sample-frac", $SampleFrac
        )

    Invoke-Step `
        -Title "2/5 - Estadísticas del Draft y Objetivos (Draft Input Analysis)" `
        -Command "python" `
        -Arguments @(
            "src/eda/draft_input_analysis.py",
            "--out-dir", (Join-Path $OutDir "draft_reports"),
            "--sample-frac", $SampleFrac
        )

    Invoke-Step `
        -Title "3/5 - Análisis Geométrico Espacial (Spatial Target Analysis)" `
        -Command "python" `
        -Arguments @(
            "src/eda/spatial_target_analysis.py",
            "--out-dir", (Join-Path $OutDir "geometry_reports"),
            "--sample-frac", $SampleFrac
        )

    Invoke-Step `
        -Title "4/5 - Estabilidad de las Etiquetas (Label Stability Analysis)" `
        -Command "python" `
        -Arguments @(
            "src/02_data_processing/02c_analyze_label_stability.py",
            "--windows", "6", "8", "10", "12", "14",
            "--labels-dir", "data/clean/labels/archive",
            "--outdir", (Join-Path $OutDir "label_stability_analysis"),
            "--sample-frac", "0.05"
        )

    Invoke-Step `
        -Title "5/5 - Generación de Figuras para el Reporte" `
        -Command "python" `
        -Arguments @(
            "src/eda/generate_report_figures.py",
            "--out-dir", (Join-Path $OutDir "report_figures")
        )

    Invoke-Step `
        -Title "6/6 - Distribución de Scores y Cuantiles" `
        -Command "python" `
        -Arguments @(
            "src/eda/plot_score_distributions.py",
            "--labels-dir", "data/clean/labels",
            "--out-dir", (Join-Path $OutDir "report_figures"),
            "--sample-frac", $SampleFrac
        )

    Write-Host "=== PIPELINE EDA COMPLETADO CON ÉXITO ===" -ForegroundColor Green
    Write-Host "Puedes revisar todos tus resultados en la carpeta: /" $OutDir -ForegroundColor Green
}
catch {
    Write-Host ""
    Write-Host "=== ERROR EN EL PIPELINE EDA ===" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}
finally {
    Write-Host ""
    Write-Host "Pipeline finalizado."
}
