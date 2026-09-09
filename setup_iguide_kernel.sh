#!/bin/bash
# Build the `mswegnn` Jupyter kernel on the I-GUIDE Platform JupyterHub (or any JupyterHub).
# Run this from a terminal on the platform:
#   bash setup_iguide_kernel.sh                 # CPU build (default; the I-GUIDE hub has no GPU)
#   bash setup_iguide_kernel.sh --cuda          # CUDA build, only if a GPU is actually present
#   bash setup_iguide_kernel.sh [--cuda] PREFIX # install somewhere other than ~/envs/mswegnn
set -euo pipefail
# pydevd prints "Debugger warning:" lines on STDOUT, which corrupts command
# substitution; the warning itself tells you to set this.
export PYDEVD_DISABLE_FILE_VALIDATION=1
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$REPO/environment-cpu.yml"
if [ "${1:-}" = "--cuda" ]; then ENV_FILE="$REPO/environment.yml"; shift; fi
ENV_PREFIX="${1:-$HOME/envs/mswegnn}"
echo "using $(basename "$ENV_FILE") -> $ENV_PREFIX"

echo "== 0. survey every kernel already installed on this hub =="
echo "   If one of them has BOTH torch_geometric AND lightning 2.0.9.post0, stop here"
echo "   and just select that kernel -- you do not need to build anything."
# Ask each kernelspec which interpreter it uses, then query that interpreter directly.
# This is more reliable than guessing conda prefixes, and it covers every kernel the hub
# offers rather than a hardcoded few.
if command -v jupyter >/dev/null 2>&1; then
  jupyter kernelspec list 2>/dev/null | awk 'NR>1 && NF>=2 {print $1" "$2}' | while read -r name dir; do
    kj="$dir/kernel.json"
    [ -f "$kj" ] || continue
    py=$(python -c "import json;print(json.load(open('$kj'))['argv'][0])" 2>/dev/null | tail -1)
    [ -n "$py" ] && [ -x "$py" ] || { printf "   %-24s (interpreter not directly runnable)\n" "$name"; continue; }
    out=$("$py" - <<'PYQ' 2>/dev/null || true
import importlib.metadata as md
got = []
for p in ("torch", "torch_geometric", "lightning"):
    try: got.append(f"{p}={md.version(p)}")
    except Exception: pass
print(" ".join(got) if got else "-")
PYQ
)
    printf "   %-24s %s\n" "$name" "${out:--}"
  done
  echo "   A dash means the kernel has none of torch / torch_geometric / lightning."
else
  echo "   jupyter not on PATH; skipping the survey"
fi

echo "== 1. free space in \$HOME (CUDA torch needs ~2.5 GB, CPU torch ~200 MB) =="
df -h "$HOME" | tail -1

echo "== 2. build the environment =="
# venv+pip by default: conda's solver gets OOM-killed on memory-limited hub containers
# ("Killed" during "Collecting package metadata"). Set USE_CONDA=1 to force conda instead.
if [ "${USE_CONDA:-0}" = "1" ]; then
  conda env create -f "$ENV_FILE" -p "$ENV_PREFIX"
else
  REQ="$REPO/requirements-cpu.txt"
  PYBIN="${PYBIN:-python}"
  ver=$("$PYBIN" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null | tail -1)
  case "$ver" in
    3.8|3.9|3.10|3.11) : ;;
    *) echo "   $PYBIN is python $ver; these wheels need 3.8-3.11."
       for alt in /cvmfs/iguide.purdue.edu/software/conda/iguide-ewd/bin/python; do
         [ -x "$alt" ] && { PYBIN="$alt"; echo "   falling back to $alt"; break; }
       done ;;
  esac
  "$PYBIN" -m venv "$ENV_PREFIX"
  "$ENV_PREFIX/bin/pip" install -q -U pip
  "$ENV_PREFIX/bin/pip" install -r "$REQ"
fi

echo "== 3. register it as a Jupyter kernel named 'mswegnn' =="
# The name must match the kernelspec recorded in test_pretrained_EN.ipynb.
"$ENV_PREFIX/bin/python" -m ipykernel install \
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
"$ENV_PREFIX/bin/python" - <<'PY'
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
