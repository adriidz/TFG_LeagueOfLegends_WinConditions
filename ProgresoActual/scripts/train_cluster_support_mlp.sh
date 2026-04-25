#!/bin/bash
#SBATCH -J new03_mlp_supv2_m11
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

PROJECT_DIR="/fhome/adiaz/TFG_LeagueOfLegends_WinConditions"
PYTHON_BIN="python3"

sleep 5
/ghome/share/example/deviceQuery || true
nvidia-smi || true

cd "${PROJECT_DIR}"

# Entorno
source "${PROJECT_DIR}/.venv_cluster/bin/activate"
# Alternativa con conda:
# source "${HOME}/miniconda3/etc/profile.d/conda.sh" && conda activate tfg

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "PROJECT_DIR=${PROJECT_DIR}"
which python3 || true
python3 --version || true

mkdir -p logs Models_new

# Input explícito: soporte v2 ya integrado en el model_input
${PYTHON_BIN} -u src/03_training/new_03_train_singleoutput_support.py \
  --input data_new/training/model_input_multioutput_regression_sample5_m11.parquet \
  --outdir Models_new/singleoutput_regression_mlp_supportv2_m11 \
  --feature-groups standard \
  --batch-size 256 \
  --epochs 60 \
  --lr 1e-3 \
  --hidden1 256 \
  --hidden2 128 \
  --dropout 0.2 \
  --weight-decay 1e-5 \
  --patience 10 \
  --val-size 0.2 \
  --seed 42

nvidia-smi || true