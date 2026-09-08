#!/bin/bash
#SBATCH --job-name=mswegnn_nb
#SBATCH --account=bcrm-tgirails
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm_notebook_%j.out
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate $CONDA_PREFIX
export WANDB_MODE=disabled
date
jupyter nbconvert --to notebook --execute --allow-errors --ExecutePreprocessor.timeout=7200 \
  --ExecutePreprocessor.kernel_name=mswegnn test_pretrained.ipynb --output test_pretrained.ipynb
echo "nbconvert EXIT=$?"; date
jupyter nbconvert --to html test_pretrained.ipynb --output plan/test_pretrained.html
python - <<'PY'
import json; nb=json.load(open('test_pretrained.ipynb'))
errs=[(i, o.get('ename'), o.get('evalue','')[:200]) for i,c in enumerate(nb['cells']) if c['cell_type']=='code' for o in c.get('outputs',[]) if o.get('output_type')=='error']
print('code cells:', sum(c['cell_type']=='code' for c in nb['cells']), '| cells with errors:', len(errs))
for e in errs: print('  cell', *e)
PY
