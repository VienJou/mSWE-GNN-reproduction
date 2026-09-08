#!/bin/bash
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate $CONDA_PREFIX
echo "START $(date)"; python plan/build_real_pickles.py; echo "EXIT=$? END $(date)"
ls -la database/datasets/train database/datasets/test
