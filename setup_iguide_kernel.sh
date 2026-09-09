#!/bin/bash
# Build the `mswegnn` Jupyter kernel on the I-GUIDE Platform JupyterHub (or any JupyterHub).
# Run this from a terminal on the platform:
#   bash setup_iguide_kernel.sh                 # CPU build (default; the I-GUIDE hub has no GPU)
#   bash setup_iguide_kernel.sh --cuda          # CUDA build, only if a GPU is actually present
#   bash setup_iguide_kernel.sh [--cuda] PREFIX # install somewhere other than ~/envs/mswegnn
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$REPO/environment-cpu.yml"
if [ "${1:-}" = "--cuda" ]; then ENV_FILE="$REPO/environment.yml"; shift; fi
ENV_PREFIX="${1:-$HOME/envs/mswegnn}"
echo "using $(basename "$ENV_FILE") -> $ENV_PREFIX"

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
conda env create -f "$ENV_FILE" -p "$ENV_PREFIX"

echo "== 3. register it as a Jupyter kernel named 'mswegnn' =="
# The name must match the kernelspec recorded in test_pretrained_EN.ipynb.
conda run -p "$ENV_PREFIX" python -m ipykernel install \
  --user --name mswegnn --display-name "Python (mswegnn)"

echo "== 4. verify =="
# non-fatal: `jupyter` may not be on PATH in a bare terminal even though the kernel is fine
if command -v jupyter >/dev/null 2>&1; then
  jupyter kernelspec list | grep -i mswegnn || echo "   not listed by jupyter; checking the path directly"
fi
ls -d "$HOME/.local/share/jupyter/kernels/mswegnn" 2>/dev/null \
  && echo "   kernelspec present" \
  || { echo "   ERROR: kernelspec was not created at ~/.local/share/jupyter/kernels/mswegnn"; exit 1; }
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
echo
echo "CPU tip: this model is fastest at ~8 threads; more can be slower. If the container"
echo "exposes many cores, run torch.set_num_threads(8) after the imports cell."
