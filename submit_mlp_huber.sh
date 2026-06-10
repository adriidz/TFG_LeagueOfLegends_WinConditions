#!/usr/bin/env bash
#SBATCH -p g2tfg11
#SBATCH --gres=shard:11000
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o mlp_huber_%j.out
#SBATCH -e mlp_huber_%j.err

echo "=== System Information ==="
hostname
date
nvidia-smi

echo "=== Activating Python Environment ==="
# Activar el entorno virtual o llamarlo directamente
# source .venv_cluster/bin/activate

echo "=== Running MLP Huber Loss Training ==="
./.venv_ultimo/bin/python3 final/scripts/04d_train_mlp_per_role_huber.py --use-wandb

echo "=== Training Finished ==="
date
