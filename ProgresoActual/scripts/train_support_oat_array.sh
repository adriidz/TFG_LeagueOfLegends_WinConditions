#!/bin/bash
#SBATCH -J support_oat
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /fhome/adiaz/TFG_LeagueOfLegends_WinConditions
#SBATCH -t 0-06:00
#SBATCH -p tfg
#SBATCH --mem 16384
#SBATCH --gres gpu:1
#SBATCH -o /fhome/adiaz/TFG_LeagueOfLegends_WinConditions/logs/%x_%u_%A_%a.out
#SBATCH -e /fhome/adiaz/TFG_LeagueOfLegends_WinConditions/logs/%x_%u_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/fhome/adiaz/TFG_LeagueOfLegends_WinConditions}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-support_oat_sample5_m12}"
MANIFEST_PATH="${MANIFEST_PATH:-ProgresoActual/experiments/support_oat/${EXPERIMENT_NAME}/runs_manifest.csv}"

cd "${PROJECT_DIR}"

VENV_ACTIVATE="${PROJECT_DIR}/.venv_cluster/bin/activate"
if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "ERROR: no existe el entorno del cluster: ${VENV_ACTIVATE}" >&2
  echo "Crea o repara .venv_cluster manualmente; este job no instala ni modifica entornos." >&2
  exit 1
fi
source "${VENV_ACTIVATE}"

mkdir -p logs ProgresoActual/models/oat_tuning ProgresoActual/cluster_run_metadata

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "ERROR: no existe MANIFEST_PATH=${MANIFEST_PATH}" >&2
  echo "Generalo en local con run_support_oat_tuning.ps1 y sincronizalo con sync_support_oat_to_cluster.ps1." >&2
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
ROW_JSON="$("${PYTHON_BIN}" -c 'import csv,json,sys; rows=list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8-sig"))); idx=int(sys.argv[2])-1; assert 0 <= idx < len(rows), f"Task id {idx+1} fuera de rango 1-{len(rows)}"; print(json.dumps(rows[idx]))' "${MANIFEST_PATH}" "${TASK_ID}")"

field() {
  "${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.argv[1]).get(sys.argv[2], ""))' "${ROW_JSON}" "$1"
}

EXPERIMENT_ID="$(field experiment_id)"
PHASE="$(field phase)"
INPUT_PATH="$(field model_input_path)"
OUTDIR="$(field train_outdir)"
SUPPORT_CONFIG_JSON="$(field support_config_json)"
FEATURE_GROUPS="$(field feature_groups)"
BATCH_SIZE="$(field batch_size)"
EPOCHS="$(field epochs)"
LR="$(field lr)"
HIDDEN1="$(field hidden1)"
HIDDEN2="$(field hidden2)"
DROPOUT="$(field dropout)"
WEIGHT_DECAY="$(field weight_decay)"
PATIENCE="$(field patience)"
VAL_SIZE="$(field val_size)"
SEED="$(field seed)"

if [[ ! -f "${INPUT_PATH}" ]]; then
  echo "ERROR: no existe model input para ${EXPERIMENT_ID}: ${INPUT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SUPPORT_CONFIG_JSON}" ]]; then
  echo "ERROR: no existe support config para ${EXPERIMENT_ID}: ${SUPPORT_CONFIG_JSON}" >&2
  exit 1
fi

read -r -a FEATURE_GROUP_ARGS <<< "${FEATURE_GROUPS}"
COMMIT_HASH="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
RUN_METADATA="ProgresoActual/cluster_run_metadata/oat_${EXPERIMENT_NAME}_${EXPERIMENT_ID}_${SLURM_JOB_ID:-nojob}_${TASK_ID}.txt"
{
  echo "experiment_name=${EXPERIMENT_NAME}"
  echo "experiment_id=${EXPERIMENT_ID}"
  echo "phase=${PHASE}"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "slurm_array_task_id=${TASK_ID}"
  echo "commit_hash=${COMMIT_HASH}"
  echo "date=$(date -Is)"
  echo "input_path=${INPUT_PATH}"
  echo "outdir=${OUTDIR}"
  echo "support_config_json=${SUPPORT_CONFIG_JSON}"
  echo "manifest_path=${MANIFEST_PATH}"
} | tee "${RUN_METADATA}"

"${PYTHON_BIN}" -u ProgresoActual/scripts/train_support_mlp_regression.py \
  --input "${INPUT_PATH}" \
  --outdir "${OUTDIR}" \
  --feature-groups "${FEATURE_GROUP_ARGS[@]}" \
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
  --support-config-json "${SUPPORT_CONFIG_JSON}" \
  --run-name "${EXPERIMENT_ID}" \
  --tags oat "${PHASE}" "${EXPERIMENT_NAME}"
