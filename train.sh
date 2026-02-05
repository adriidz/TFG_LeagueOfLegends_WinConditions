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

source /fhome/maed01/tfg/.venv/bin/activate

python -u -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('cuda_device_count=', torch.cuda.device_count())"
# opcional (si el nodo tiene nvidia-smi):
# nvidia-smi || true

srun python -u Training/train.py