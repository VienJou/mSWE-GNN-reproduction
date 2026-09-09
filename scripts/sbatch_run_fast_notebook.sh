#!/bin/bash
#SBATCH --job-name=fastnb
#SBATCH --partition=gpu_a100,gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --gres=gpu:1 --time=1:00:00
#SBATCH --output=/projects/bcrm/wz53/Aging_dam/mSWE-GNN-main/plan/fastnb_%j.log
#SBATCH --error=/projects/bcrm/wz53/Aging_dam/mSWE-GNN-main/plan/fastnb_%j.err
set -e
source /u/wz53/miniconda3/etc/profile.d/conda.sh
conda activate mswegnn
cd /projects/bcrm/wz53/Aging_dam/mSWE-GNN-main
echo "NODE=$(hostname) GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader) START=$(date)"
export WANDB_MODE=disabled PYDEVD_DISABLE_FILE_VALIDATION=1
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=mswegnn \
  --ExecutePreprocessor.timeout=-1 \
  test_pretrained_EN_fast.ipynb
echo "END=$(date)"
