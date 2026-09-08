#!/bin/bash
# Unzip the Zenodo archives and build the real pickles exactly like database/create_dataset.ipynb.
# Usage: bash plan/prepare_real_data.sh        (run after both wget downloads finish)
set -e
R=.
cd $R/database
for z in raw_datasets_mesh raw_datasets_dk15; do
  [ -f $z.zip ] || { echo "missing $z.zip"; exit 1; }
  echo "== $z.zip: $(stat -c %s $z.zip) bytes"; unzip -tq $z.zip | tail -1
  mkdir -p _unz_$z && unzip -q -o $z.zip -d _unz_$z
  # archives may contain a top-level folder; locate the dir that has Simulations/
  src=$(find _unz_$z -maxdepth 3 -type d -name Simulations | head -1); src=$(dirname "$src")
  [ -n "$src" ] || { echo "no Simulations/ inside $z.zip"; find _unz_$z -maxdepth 2 | head; exit 1; }
  rsync -a "$src"/ $z/ && rm -rf _unz_$z
  echo "   $z: $(ls $z/Simulations | wc -l) nc, $(ls $z/DEM | wc -l) DEM, $(ls $z/Hydrograph | wc -l) hydrographs, $(ls $z/Geometry | wc -l) polygons"
done
cd $R
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate $CONDA_PREFIX
python - <<'PY'
import os, sys, time
sys.path.insert(0, os.getcwd())
from database.graph_creation import create_mesh_dataset, save_database, create_dataset_folders
create_dataset_folders('database/datasets')
# identical to database/create_dataset.ipynb (names without the trailing '2' so config.yaml finds them)
jobs = [
    ['database/raw_datasets_mesh', 'multiscale_mesh_dataset', 'database/datasets/train', True, 1, 80],
    ['database/raw_datasets_mesh', 'multiscale_mesh_dataset', 'database/datasets/test',  True, 81, 20],
    ['database/raw_datasets_mesh', 'mesh_dataset',            'database/datasets/train', False, 1, 80],
    ['database/raw_datasets_mesh', 'mesh_dataset',            'database/datasets/test',  False, 81, 20],
    ['database/raw_datasets_dk15', 'dijkring_15',             'database/datasets/test',  True, 101, 11],
]
for folder, name, out, ms, start, n in jobs:
    t0 = time.time()
    ds = create_mesh_dataset(folder, n, start, with_multiscale=ms, number_of_multiscales=4)
    if name.startswith('dijkring'):
        save_database(ds[:1], name=name, out_path='database/datasets/train')
        save_database(ds[1:], name=name, out_path='database/datasets/test')
    else:
        save_database(ds, name=name, out_path=out)
    print(f'{name} {out} n={n} done in {time.time()-t0:.0f}s: {ds[0]}')
PY
ls -la database/datasets/train database/datasets/test
