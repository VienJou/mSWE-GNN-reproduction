#!/bin/bash
#SBATCH --job-name=mswegnn_real
#SBATCH --account=bcrm-tgirails
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm_gpu_real_%j.out
# 1) paper's pretrained K4_F64 on the REAL test set (config.yaml unchanged) -> compare to paper Table
# 2) 2-epoch training at full model size on the real train set (plan/config_real_smoke.yaml)
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_PREFIX
export WANDB_MODE=offline
nvidia-smi -L
echo "=== [1] test_model.py, pretrained K4_F64.h5, real test set ==="; date
python plan/run_test_smoke.py config.yaml
echo "=== [2] main.py 2-epoch train/val/test, real data ==="; date
python plan/run_main_smoke.py plan/config_real_smoke.yaml
echo "EXIT=$?"; date
