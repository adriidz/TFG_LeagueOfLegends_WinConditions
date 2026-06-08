#!/usr/bin/env bash
#SBATCH -p g2tfg11,g2tfg12
#SBATCH --gres=shard:11000
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -o mlp_huber_%j.out
#SBATCH -e mlp_huber_%j.err

echo "=== System Information ==="
hostname
date
nvidia-smi

echo "=== Activating Python Environment ==="
# Si el cluster requiere activar el entorno virtual, descomenta la siguiente línea 
# (o cámbiala por la ruta a tu entorno conda/virtualenv)
# source .venv/bin/activate

echo "=== Running MLP Huber Loss Training ==="
python final/scripts/04d_train_mlp_per_role_huber.py --use-wandb

echo "=== Training Finished ==="
date
