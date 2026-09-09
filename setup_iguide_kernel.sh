#!/bin/bash
# Build the `mswegnn` Jupyter kernel on the I-GUIDE Platform JupyterHub (or any JupyterHub).
# Run this from a terminal on the platform:  bash setup_iguide_kernel.sh [env_prefix]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${1:-$HOME/envs/mswegnn}"

echo "== 0. what the stock kernels already contain (read-only, on CVMFS) =="
echo "   If one of them already has lightning==2.0.9.post0 AND torch_geometric,"
echo "   you can skip the rest of this script and just select that kernel."
for k in geoai iguide iguide-ewd; do
  p="/cvmfs/iguide.purdue.edu/software/conda/$k"
  if [ -d "$p" ]; then
    echo "--- $k ---"
    conda env export -p "$p" 2>/dev/null \
      | grep -iE "^\s*-\s*(python|torch|torch-geometric|torch_geometric|lightning|pytorch)[=<>: ]" || echo "   (none of the relevant packages)"
  fi
done

echo "== 1. free space in \$HOME (CUDA torch needs ~2.5 GB, CPU torch ~200 MB) =="
df -h "$HOME" | tail -1

echo "== 2. build the environment =="
conda env create -f "$REPO/environment.yml" -p "$ENV_PREFIX"

echo "== 3. register it as a Jupyter kernel named 'mswegnn' =="
# The name must match the kernelspec recorded in test_pretrained_EN.ipynb.
conda run -p "$ENV_PREFIX" python -m ipykernel install \
  --user --name mswegnn --display-name "Python (mswegnn)"

echo "== 4. verify =="
jupyter kernelspec list | grep -i mswegnn
cd "$REPO"
conda run -p "$ENV_PREFIX" python - <<'PY'
import sys, os, importlib.metadata as md
sys.path.insert(0, os.getcwd())
from utils.visualization import *
from models.gnn import *
from training.train import *
from database.graph_creation import *
import torch
print("import chain OK | python", sys.version.split()[0],
      "| torch", md.version("torch"), "| PyG", md.version("torch_geometric"),
      "| lightning", md.version("lightning"), "| cuda available:", torch.cuda.is_available())
PY
echo
echo "Done. Reload the browser tab, open test_pretrained_EN.ipynb, and pick 'Python (mswegnn)'."
echo "The notebook also needs database/datasets/{train,test}/*.pkl - see README."
