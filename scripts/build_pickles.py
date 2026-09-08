"""Convert database/raw_datasets_synth -> database/datasets_synth/{train,test}/multiscale_mesh_dataset.pkl
Same call chain as database/create_dataset.ipynb. Run from repo root in the mswegnn env."""
import os, sys, time
sys.path.insert(0, os.getcwd())
from database.graph_creation import create_mesh_dataset, save_database, create_dataset_folders

RAW = sys.argv[1] if len(sys.argv) > 1 else "database/raw_datasets_synth"
OUT = sys.argv[2] if len(sys.argv) > 2 else "database/datasets_synth"
N_TRAIN = int(sys.argv[3]) if len(sys.argv) > 3 else 4
N_TEST = int(sys.argv[4]) if len(sys.argv) > 4 else 2

create_dataset_folders(OUT)
t0 = time.time()
train = create_mesh_dataset(RAW, N_TRAIN, 1, with_multiscale=True, number_of_multiscales=4)
save_database(train, name="multiscale_mesh_dataset", out_path=f"{OUT}/train")
test = create_mesh_dataset(RAW, N_TEST, 1 + N_TRAIN, with_multiscale=True, number_of_multiscales=4)
save_database(test, name="multiscale_mesh_dataset", out_path=f"{OUT}/test")
d = train[0]
print(f"built {len(train)} train / {len(test)} test in {time.time()-t0:.1f}s")
print("sample:", d)
print("mesh:", d.mesh, "| node_ptr:", d.node_ptr.tolist(), "| WD:", tuple(d.WD.shape), "| BC:", tuple(d.BC.shape), "| node_BC:", d.node_BC.tolist())
