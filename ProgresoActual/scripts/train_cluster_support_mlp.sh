#!/bin/bash
#SBATCH -J support_mlp_m12
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /fhome/adiaz/TFG_LeagueOfLegends_WinConditions
#SBATCH -t 0-06:00
#SBATCH -p tfg
#SBATCH --mem 16384
#SBATCH --gres gpu:1
#SBATCH -o /fhome/adiaz/TFG_LeagueOfLegends_WinConditions/logs/%x_%u_%j.out
#SBATCH -e /fhome/adiaz/TFG_LeagueOfLegends_WinConditions/logs/%x_%u_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/fhome/adiaz/TFG_LeagueOfLegends_WinConditions}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

sleep 5
/ghome/share/example/deviceQuery || true
nvidia-smi || true

cd "${PROJECT_DIR}"

VENV_ACTIVATE="${PROJECT_DIR}/.venv_cluster/bin/activate"
if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "ERROR: no existe el entorno del cluster: ${VENV_ACTIVATE}" >&2
  echo "Crea o repara .venv_cluster manualmente en el cluster; este job no instala dependencias ni modifica entornos." >&2
  exit 1
fi

source "${VENV_ACTIVATE}"
# Alternativa con conda:
# source "${HOME}/miniconda3/etc/profile.d/conda.sh" && conda activate tfg

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "PROJECT_DIR=${PROJECT_DIR}"
which python3 || true
python3 --version || true

mkdir -p logs ProgresoActual/models ProgresoActual/cluster_run_metadata

# Defaults aligned with the local preparation + sync flow.
# Override any of these when submitting, e.g.:
#   SAMPLE_TAG=sample5 WINDOW_TAG=m12 EPOCHS=80 sbatch ProgresoActual/scripts/train_cluster_support_mlp.sh
SAMPLE_TAG="${SAMPLE_TAG:-sample5}"
WINDOW_TAG="${WINDOW_TAG:-m12}"
RUN_NAME="${RUN_NAME:-support_mlp_${SAMPLE_TAG}_${WINDOW_TAG}}"

if [[ -z "${SAMPLE_TAG}" || "${SAMPLE_TAG}" == "full" ]]; then
  SAMPLE_SUFFIX=""
  DEFAULT_INPUT_PATH="ProgresoActual/data/training/model_input_support_regression_${WINDOW_TAG}.parquet"
  LEGACY_FULL_INPUT_PATH="ProgresoActual/data/training/model_input_support_regression_full_${WINDOW_TAG}.parquet"
else
  SAMPLE_SUFFIX="_${SAMPLE_TAG}"
  DEFAULT_INPUT_PATH="ProgresoActual/data/training/model_input_support_regression${SAMPLE_SUFFIX}_${WINDOW_TAG}.parquet"
  LEGACY_FULL_INPUT_PATH=""
fi

INPUT_PATH="${INPUT_PATH:-${DEFAULT_INPUT_PATH}}"
if [[ "${SAMPLE_TAG}" == "full" && "${INPUT_PATH}" == "${LEGACY_FULL_INPUT_PATH}" && -f "${DEFAULT_INPUT_PATH}" ]]; then
  echo "WARN: INPUT_PATH usaba el nombre antiguo para full (${LEGACY_FULL_INPUT_PATH}). Uso el artefacto canonico: ${DEFAULT_INPUT_PATH}" >&2
  INPUT_PATH="${DEFAULT_INPUT_PATH}"
fi
OUTDIR="${OUTDIR:-ProgresoActual/models/${RUN_NAME}}"
SUPPORT_CONFIG_JSON="${SUPPORT_CONFIG_JSON:-ProgresoActual/data/clean/scores/selected_support_score_config.json}"

FEATURE_GROUPS="${FEATURE_GROUPS:-standard}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-60}"
LR="${LR:-1e-3}"
HIDDEN1="${HIDDEN1:-256}"
HIDDEN2="${HIDDEN2:-128}"
DROPOUT="${DROPOUT:-0.2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
PATIENCE="${PATIENCE:-10}"
VAL_SIZE="${VAL_SIZE:-0.2}"
SEED="${SEED:-42}"

COMMIT_HASH="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
RUN_METADATA="ProgresoActual/cluster_run_metadata/${RUN_NAME}_${SLURM_JOB_ID:-nojob}.txt"
{
  echo "run_name=${RUN_NAME}"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "commit_hash=${COMMIT_HASH}"
  echo "hostname=$(hostname)"
  echo "date=$(date -Is)"
  echo "input_path=${INPUT_PATH}"
  echo "outdir=${OUTDIR}"
  echo "support_config_json=${SUPPORT_CONFIG_JSON}"
  echo "feature_groups=${FEATURE_GROUPS}"
  echo "batch_size=${BATCH_SIZE}"
  echo "epochs=${EPOCHS}"
  echo "lr=${LR}"
  echo "hidden1=${HIDDEN1}"
  echo "hidden2=${HIDDEN2}"
  echo "dropout=${DROPOUT}"
  echo "weight_decay=${WEIGHT_DECAY}"
  echo "patience=${PATIENCE}"
  echo "val_size=${VAL_SIZE}"
  echo "seed=${SEED}"
} | tee "${RUN_METADATA}"

if [[ ! -f "${INPUT_PATH}" ]]; then
  echo "ERROR: INPUT_PATH no existe: ${INPUT_PATH}" >&2
  if [[ "${SAMPLE_TAG}" == "full" ]]; then
    echo "Para SAMPLE_TAG=full el input canonico no lleva sufijo _full:" >&2
    echo "  ${DEFAULT_INPUT_PATH}" >&2
    if [[ "${INPUT_PATH}" == "${LEGACY_FULL_INPUT_PATH}" ]]; then
      echo "El path con _full es antiguo/incorrecto para este flujo:" >&2
      echo "  ${LEGACY_FULL_INPUT_PATH}" >&2
    fi
  fi
  echo "Generalo primero en local con:" >&2
  if [[ "${SAMPLE_TAG}" == "full" ]]; then
    echo "  .\\ProgresoActual\\run_support_pipeline.ps1 -SampleFrac 1 -SkipFrameState -SkipDraftFeatures" >&2
  else
    echo "  .\\ProgresoActual\\run_support_pipeline.ps1 -SampleFrac 0.05" >&2
  fi
  echo "y copialo al cluster con:" >&2
  echo "  .\\ProgresoActual\\scripts\\sync_support_artifacts_to_cluster.ps1 -SampleTag ${SAMPLE_TAG} -WindowTag ${WINDOW_TAG}" >&2
  echo "Default esperado para este job: ${DEFAULT_INPUT_PATH}" >&2
  exit 1
fi

if [[ ! -f "${SUPPORT_CONFIG_JSON}" ]]; then
  echo "WARN: no existe SUPPORT_CONFIG_JSON=${SUPPORT_CONFIG_JSON}; entreno sin config de heuristica." >&2
  SUPPORT_CONFIG_ARG=()
else
  SUPPORT_CONFIG_ARG=(--support-config-json "${SUPPORT_CONFIG_JSON}")
fi

"${PYTHON_BIN}" -u ProgresoActual/scripts/train_support_mlp_regression.py \
  --input "${INPUT_PATH}" \
  --outdir "${OUTDIR}" \
  --feature-groups "${FEATURE_GROUPS}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --hidden1 "${HIDDEN1}" \
  --hidden2 "${HIDDEN2}" \
  --dropout "${DROPOUT}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --patience "${PATIENCE}" \
  --val-size "${VAL_SIZE}" \
  --seed "${SEED}" \
  "${SUPPORT_CONFIG_ARG[@]}"

nvidia-smi || true
