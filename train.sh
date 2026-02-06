#!/bin/bash
#SBATCH --job-name=lol_train
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH -t 4-00:05
#SBATCH -p tfg
#SBATCH --gres=gpu:1
#SBATCH -D /fhome/maed01/tfg
#SBATCH -o /fhome/maed01/tfg/Joblists/slurm-%x-%j.out
#SBATCH -e /fhome/maed01/tfg/Joblists/slurm-%x-%j.err

set -euo pipefail

# Por defecto 'both', pero se puede pasar como argumento:
#   sbatch train.sh both
#   sbatch train.sh emb_only
#   sbatch train.sh pca_only
MODE=${1:-both}

source /fhome/maed01/tfg/.venv/bin/activate

python -u -c "import torch; print('cuda_available=', torch.cuda.is_available())"

echo "=== Ejecutando experimento: $MODE ==="
srun python -u Training/inference.py --mode "$MODE"