#!/bin/bash
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_PREFIX
export WANDB_MODE=offline OMP_NUM_THREADS=2
echo "START $(date)"
python plan/run_main_smoke.py plan/config_smoke.yaml
echo "EXIT=$? END $(date)"
