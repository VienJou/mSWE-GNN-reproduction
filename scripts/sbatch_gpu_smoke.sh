#!/bin/bash
#SBATCH --job-name=mswegnn_smoke
#SBATCH --account=bcrm-tgirails
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_gpu_smoke_%j.out
# Same train->val->test smoke as the CPU run, but on one H100.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_PREFIX
export WANDB_MODE=offline
nvidia-smi -L
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
CFG=${1:-plan/config_smoke.yaml}
echo "=== TRAIN/VAL/TEST via main.py with $CFG ==="
python plan/run_main_smoke.py "$CFG"
