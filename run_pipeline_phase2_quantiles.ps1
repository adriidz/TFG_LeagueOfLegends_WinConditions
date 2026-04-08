$ErrorActionPreference = "Stop"

# ── CONFIGURACIÓN GENERAL ─────────────────────────────────────────────────────
$SampleFrac = "1.0"
$LabelMaxMinute = "10"
$LabelSchema = "binary_clean"
$FeatureGroups = "standard"

# Configuraciones de quantiles a probar
$QuantileConfigs = @(
    @{ Lower = "0.20"; Upper = "0.80" },
    @{ Lower = "0.30"; Upper = "0.70" },
    @{ Lower = "0.40"; Upper = "0.60" }
)

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

    & $Command @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw ("El comando fallo con exit code " + $LASTEXITCODE + "`n" + $Command + " " + ($Arguments -join " "))
    }

    Write-Host ("OK - " + $Title) -ForegroundColor Green
}

function Get-QTag {
    param(
        [string]$Lower,
        [string]$Upper
    )

    $l = [int]([double]$Lower * 100)
    $u = [int]([double]$Upper * 100)
    return ("q" + "{0:D2}" -f $l) + "_" + ("{0:D2}" -f $u)
}

try {
    Write-Host "=== INICIO FASE 2 QUANTILES (LOCAL: SOLO CPU) ===" -ForegroundColor Yellow
    Write-Host ("Sample frac      : " + $SampleFrac)
    Write-Host ("Label max minute : " + $LabelMaxMinute)
    Write-Host ("Label schema     : " + $LabelSchema)
    Write-Host ("Feature groups   : " + $FeatureGroups)
    Write-Host ("Directorio actual: " + (Get-Location).Path)
    Write-Host ""

    foreach ($cfg in $QuantileConfigs) {
        $LowerQ = $cfg.Lower
        $UpperQ = $cfg.Upper
        $QTag = Get-QTag -Lower $LowerQ -Upper $UpperQ

        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Yellow
        Write-Host ("EXPERIMENTO ACTUAL: " + $QTag) -ForegroundColor Yellow
        Write-Host "============================================================" -ForegroundColor Yellow

        Invoke-Step `
            -Title ("1/2 - Construyendo labels + draft features (" + $QTag + ")") `
            -Command "python" `
            -Arguments @(
                "src/02_data_processing/02a_p3_build_labels_and_draft_features.py",
                "--analysis-max-minutes", $LabelMaxMinute,
                "--label-schema", $LabelSchema,
                "--lower-quantile", $LowerQ,
                "--upper-quantile", $UpperQ,
                "--sample-frac", $SampleFrac
            )

        Invoke-Step `
            -Title ("2/2 - Construyendo model input (" + $QTag + ")") `
            -Command "python" `
            -Arguments @(
                "src/02_data_processing/02b_p3_build_model_input.py",
                "--label-max-minute", $LabelMaxMinute,
                "--label-schema", $LabelSchema,
                "--lower-quantile", $LowerQ,
                "--upper-quantile", $UpperQ,
                "--sample-frac", $SampleFrac
            )

        Write-Host ""
        Write-Host ("Experimento local completado: " + $QTag) -ForegroundColor Green
        Write-Host "Comando para lanzar en Kaggle:" -ForegroundColor Yellow
        Write-Host (
            "python /kaggle/working/TFG/src/03_training/03_p3_train_multioutput.py " +
            "--feature-groups " + $FeatureGroups + " " +
            "--sample-frac " + $SampleFrac + " " +
            "--label-max-minute " + $LabelMaxMinute + " " +
            "--target-schema " + $LabelSchema + " " +
            "--lower-quantile " + $LowerQ + " " +
            "--upper-quantile " + $UpperQ
        ) -ForegroundColor White

        Write-Host ""
    }

    Write-Host "=== FASE 2 QUANTILES LOCAL COMPLETADA ===" -ForegroundColor Green
    Write-Host "Ya puedes subir a Kaggle los model_input y ejecutar solo el entrenamiento con GPU." -ForegroundColor Green
}
catch {
    Write-Host ""
    Write-Host "=== ERROR EN EL PIPELINE ===" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}
finally {
    Write-Host ""
    Write-Host "Pulsa Enter para cerrar..."
    Read-Host
}