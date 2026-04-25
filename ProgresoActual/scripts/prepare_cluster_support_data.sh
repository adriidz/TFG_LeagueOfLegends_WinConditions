#!/bin/bash
#SBATCH -J prep_support_m12
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /fhome/adiaz/TFG_LeagueOfLegends_WinConditions
#SBATCH -t 0-06:00
#SBATCH -p tfg
#SBATCH --mem 32768
#SBATCH -o /fhome/adiaz/TFG_LeagueOfLegends_WinConditions/logs/%x_%u_%j.out
#SBATCH -e /fhome/adiaz/TFG_LeagueOfLegends_WinConditions/logs/%x_%u_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/fhome/adiaz/TFG_LeagueOfLegends_WinConditions}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${PROJECT_DIR}"

source "${PROJECT_DIR}/.venv_cluster/bin/activate"
# Alternativa con conda:
# source "${HOME}/miniconda3/etc/profile.d/conda.sh" && conda activate tfg

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "PROJECT_DIR=${PROJECT_DIR}"
which python3 || true
python3 --version || true

mkdir -p logs \
  ProgresoActual/data/clean/frame_state \
  ProgresoActual/data/clean/features \
  ProgresoActual/data/clean/scores \
  ProgresoActual/data/training \
  ProgresoActual/analysis/support_grid \
  ProgresoActual/analysis/support_label_distribution \
  ProgresoActual/cluster_run_metadata

# Defaults for the first clean support-only benchmark.
# Override any of these when submitting, e.g.:
#   SAMPLE_FRAC=0.10 SAMPLE_TAG=sample10 sbatch ProgresoActual/scripts/prepare_cluster_support_data.sh
RAW_ROOT="${RAW_ROOT:-data/raw/raw}"
REGION="${REGION:-europe}"
SAMPLE_FRAC="${SAMPLE_FRAC:-0.05}"
SAMPLE_TAG="${SAMPLE_TAG:-sample5}"
MAX_MATCHES="${MAX_MATCHES:-0}"
SEED="${SEED:-42}"

WINDOW_MINUTE="${WINDOW_MINUTE:-12}"
WINDOW_TAG="${WINDOW_TAG:-m12}"
START_MINUTES="${START_MINUTES:-5}"
MAX_MINUTES="${MAX_MINUTES:-${WINDOW_MINUTE}}"
FAR_ADC_THRESHOLDS="${FAR_ADC_THRESHOLDS:-2500}"
WEIGHT_TRIPLETS="${WEIGHT_TRIPLETS:-0.45,0.35,0.20}"
EXPORT_BEST="${EXPORT_BEST:-coverage}"

SAMPLE_ARGS=()
if [[ "${SAMPLE_FRAC}" != "0" && "${SAMPLE_FRAC}" != "0.0" && "${SAMPLE_FRAC}" != "1" && "${SAMPLE_FRAC}" != "1.0" ]]; then
  SAMPLE_ARGS=(--sample-frac "${SAMPLE_FRAC}")
fi

SAMPLE_SUFFIX=""
if [[ ${#SAMPLE_ARGS[@]} -gt 0 ]]; then
  SAMPLE_SUFFIX="_${SAMPLE_TAG}"
fi

FRAME_STATE_DIR="${FRAME_STATE_DIR:-ProgresoActual/data/clean/frame_state}"
FRAME_STATE_NAME="${FRAME_STATE_NAME:-support_frame_state}"
DRAFT_DIR="${DRAFT_DIR:-ProgresoActual/data/clean/features}"
DRAFT_NAME="${DRAFT_NAME:-draft_features}"

FRAME_STATE_PATH="${FRAME_STATE_PATH:-${FRAME_STATE_DIR}/${FRAME_STATE_NAME}${SAMPLE_SUFFIX}.parquet}"
DRAFT_PATH="${DRAFT_PATH:-${DRAFT_DIR}/${DRAFT_NAME}${SAMPLE_SUFFIX}.parquet}"
SUPPORT_SCORES_PATH="${SUPPORT_SCORES_PATH:-ProgresoActual/data/clean/scores/support_scores${SAMPLE_SUFFIX}_${WINDOW_TAG}.parquet}"
MODEL_INPUT_PATH="${MODEL_INPUT_PATH:-ProgresoActual/data/training/model_input_support_regression${SAMPLE_SUFFIX}_${WINDOW_TAG}.parquet}"
LABEL_DISTRIBUTION_DIR="${LABEL_DISTRIBUTION_DIR:-ProgresoActual/analysis/support_label_distribution/${SAMPLE_TAG}_${WINDOW_TAG}}"

MAX_MATCH_ARGS=()
if [[ "${MAX_MATCHES}" != "0" ]]; then
  MAX_MATCH_ARGS=(--max-matches "${MAX_MATCHES}")
fi

read -r -a START_MINUTES_ARGS <<< "${START_MINUTES}"
read -r -a MAX_MINUTES_ARGS <<< "${MAX_MINUTES}"
read -r -a FAR_ADC_THRESHOLDS_ARGS <<< "${FAR_ADC_THRESHOLDS}"
read -r -a WEIGHT_TRIPLETS_ARGS <<< "${WEIGHT_TRIPLETS}"

COMMIT_HASH="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
RUN_METADATA="ProgresoActual/cluster_run_metadata/prepare_${SAMPLE_TAG}_${WINDOW_TAG}_${SLURM_JOB_ID:-nojob}.txt"
{
  echo "job=prepare_cluster_support_data"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "commit_hash=${COMMIT_HASH}"
  echo "hostname=$(hostname)"
  echo "date=$(date -Is)"
  echo "raw_root=${RAW_ROOT}"
  echo "region=${REGION}"
  echo "sample_frac=${SAMPLE_FRAC}"
  echo "sample_tag=${SAMPLE_TAG}"
  echo "max_matches=${MAX_MATCHES}"
  echo "seed=${SEED}"
  echo "window_minute=${WINDOW_MINUTE}"
  echo "window_tag=${WINDOW_TAG}"
  echo "start_minutes=${START_MINUTES}"
  echo "max_minutes=${MAX_MINUTES}"
  echo "far_adc_thresholds=${FAR_ADC_THRESHOLDS}"
  echo "weight_triplets=${WEIGHT_TRIPLETS}"
  echo "export_best=${EXPORT_BEST}"
  echo "frame_state_path=${FRAME_STATE_PATH}"
  echo "draft_path=${DRAFT_PATH}"
  echo "support_scores_path=${SUPPORT_SCORES_PATH}"
  echo "model_input_path=${MODEL_INPUT_PATH}"
  echo "label_distribution_dir=${LABEL_DISTRIBUTION_DIR}"
} | tee "${RUN_METADATA}"

run_step() {
  local name="$1"
  shift
  local start_ts
  local end_ts
  start_ts="$(date +%s)"
  echo ""
  echo "==== ${name} ===="
  echo "$*"
  "$@"
  end_ts="$(date +%s)"
  echo "==== ${name} finished in $((end_ts - start_ts))s ===="
}

run_step "Build draft features" \
  "${PYTHON_BIN}" -u ProgresoActual/src/02_data_processing/build_draft_features.py \
  --raw-root "${RAW_ROOT}" \
  --region "${REGION}" \
  --outdir "${DRAFT_DIR}" \
  --out-name "${DRAFT_NAME}" \
  --seed "${SEED}" \
  "${SAMPLE_ARGS[@]}" \
  "${MAX_MATCH_ARGS[@]}"

run_step "Extract support frame state" \
  "${PYTHON_BIN}" -u ProgresoActual/src/02_data_processing/new_02a_extract_support_frame_state.py \
  --raw-root "${RAW_ROOT}" \
  --region "${REGION}" \
  --outdir "${FRAME_STATE_DIR}" \
  --out-name "${FRAME_STATE_NAME}" \
  --seed "${SEED}" \
  "${SAMPLE_ARGS[@]}" \
  "${MAX_MATCH_ARGS[@]}"

run_step "Grid/export support scores" \
  "${PYTHON_BIN}" -u ProgresoActual/src/02_data_processing/new_02b_grid_support_scores.py \
  --frame-state-dir "${FRAME_STATE_DIR}" \
  --frame-state-name "${FRAME_STATE_NAME}" \
  --outdir ProgresoActual/analysis/support_grid \
  --start-minutes "${START_MINUTES_ARGS[@]}" \
  --max-minutes "${MAX_MINUTES_ARGS[@]}" \
  --far-adc-thresholds "${FAR_ADC_THRESHOLDS_ARGS[@]}" \
  --weight-triplets "${WEIGHT_TRIPLETS_ARGS[@]}" \
  --champion-summary \
  --export-best "${EXPORT_BEST}" \
  --export-support-scores-path "${SUPPORT_SCORES_PATH}" \
  --write-config-json \
  "${SAMPLE_ARGS[@]}"

run_step "Build support model input" \
  "${PYTHON_BIN}" -u ProgresoActual/src/02_data_processing/build_support_model_input.py \
  --draft-path "${DRAFT_PATH}" \
  --support-scores-path "${SUPPORT_SCORES_PATH}" \
  --out-path "${MODEL_INPUT_PATH}"

run_step "Plot support label distribution" \
  "${PYTHON_BIN}" -u ProgresoActual/scripts/plot_support_label_distribution.py \
  --support-scores-path "${SUPPORT_SCORES_PATH}" \
  --outdir "${LABEL_DISTRIBUTION_DIR}"

echo ""
echo "Preparation finished."
echo "Frame state:    ${FRAME_STATE_PATH}"
echo "Draft features: ${DRAFT_PATH}"
echo "Support scores: ${SUPPORT_SCORES_PATH}"
echo "Model input:    ${MODEL_INPUT_PATH}"
echo "Label plots:    ${LABEL_DISTRIBUTION_DIR}"
echo "Next step:      sbatch ProgresoActual/scripts/train_cluster_support_mlp.sh"
