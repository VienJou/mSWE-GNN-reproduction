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
