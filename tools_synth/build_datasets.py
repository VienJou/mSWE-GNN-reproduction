"""Script version of database/create_dataset.ipynb, pointed at the synthetic
raw dataset. Produces database/datasets/{train,test}/<name>.pkl
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from database.graph_creation import (  # noqa: E402
    create_mesh_dataset, save_database, create_dataset_folders)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=os.path.join(REPO, "database", "raw_datasets_synth"))
    p.add_argument("--datasets", default=os.path.join(REPO, "database", "datasets_synth"))
    p.add_argument("--name", default="multiscale_mesh_dataset")
    p.add_argument("--train-start", type=int, default=1)
    p.add_argument("--train-n", type=int, default=6)
    p.add_argument("--test-start", type=int, default=81)
    p.add_argument("--test-n", type=int, default=2)
    p.add_argument("--n-scales", type=int, default=4)
    p.add_argument("--multiscale", action="store_true", default=True)
    p.add_argument("--single-scale", dest="multiscale", action="store_false")
    args = p.parse_args()

    create_dataset_folders(dataset_folder=args.datasets)

    for split, start, n in (("train", args.train_start, args.train_n),
                            ("test", args.test_start, args.test_n)):
        print(f"[{split}] building {n} simulations from id {start}", flush=True)
        ds = create_mesh_dataset(args.raw, n, start,
                                 with_multiscale=args.multiscale,
                                 number_of_multiscales=args.n_scales)
        out = os.path.join(args.datasets, split)
        save_database(ds, name=args.name, out_path=out)
        print(f"[{split}] saved {len(ds)} sims -> {out}/{args.name}.pkl", flush=True)
        d = ds[0]
        print(f"[{split}] sample: num_nodes={d.num_nodes} "
              f"edges={d.edge_index.shape[1]} WD={tuple(d.WD.shape)} "
              f"scales={getattr(d.mesh, 'num_meshes', 1)}", flush=True)


if __name__ == "__main__":
    main()
