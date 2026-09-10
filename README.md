# mSWE-GNN — paper walkthrough and pretrained-model reproduction

A study fork of **mSWE-GNN**, the multi-scale hydraulic graph neural network for flood
modelling. It adds a self-contained Jupyter notebook that reads the paper and the code
side by side, and that reproduces the published test-set results from the pretrained
checkpoints without any training.

| | |
|---|---|
| Paper | Bentivoglio, Isufi, Jonkman & Taormina, *Multi-scale hydraulic graph neural networks for flood modelling*, **Nat. Hazards Earth Syst. Sci. 25, 335–351, 2025**, [doi:10.5194/nhess-25-335-2025](https://doi.org/10.5194/nhess-25-335-2025) (open access, CC BY 4.0) |
| Upstream code | [github.com/RBentivoglio/mSWE-GNN](https://github.com/RBentivoglio/mSWE-GNN) — v1.1, Nov 2024, MIT licence |
| Simulation data | Zenodo [10.5281/zenodo.13326595](https://doi.org/10.5281/zenodo.13326595) (not redistributed here, see below) |

## What is in this repository

**`test_pretrained_EN.ipynb`** — the main deliverable. It was executed on 1×H100 and all 145
outputs and figures are saved, so it can be read end to end without running anything. Its 61
cells interleave two layers of content in the order the paper is written:

1. **Paper walkthrough** — the research problem and the four gaps of the earlier SWE-GNN,
   the method (multi-scale mesh and graph, encoder/processor/decoder, ghost-cell boundary
   conditions, rotation-invariant inputs, loss and curriculum learning), the experimental
   setup, the results and the discussion. The paper's own figures are embedded from the
   open-access article (see *Attribution* below).
2. **Reproduction** — loading the 17 pretrained checkpoints shipped with the upstream repo,
   rolling them out autoregressively on the real test sets, and reproducing the global
   metrics, the per-simulation ranking, the single-simulation visualisations, the
   speed-versus-accuracy Pareto front over all 16 models, and the dike-ring-15 transfer case.

Section 2.2.1 adds an analysis that is not in the paper: the receptive field and compute
cost of the multi-scale U path, measured from the actual meshes (hop distances of
91 / 185 / 371 / 739 m across the four scales, giving an 88·h₁ ≈ 8.1 km single-pass
receptive field for about 10.6 finest-scale-layer equivalents of compute).

**`test_pretrained_EN_fast.ipynb`** — the same walkthrough, trimmed to run in about
**8 minutes on a CPU-only container** instead of roughly 4 hours, for demonstrating the
workflow live. Three changes, each marked `FAST VARIANT` in the code with the original line
left commented directly above it: 3 synthetic test simulations instead of 20, the
16-checkpoint Pareto sweep skipped (the figure is still drawn from the authors' recorded
csv), and 2 dike-ring-15 simulations instead of 10. **Its metrics are computed over 3
simulations and will not match the paper** — quote the full notebook, run this one. 19 of the
22 code cells are byte-identical between the two.

It ships with outputs from one H100 run so it can also be read as-is, but those timings are
GPU timings: 0.47 s per simulation there against roughly 40 s on a CPU container. The
accuracy figures are hardware-independent; re-running replaces both.

**Also included**: the upstream source (`models/`, `utils/`, `training/`, `database/*.py`),
the reference configs, all 17 pretrained checkpoints (`results/`), the authors' recorded
result tables (`*.csv`), the paper figures used in the notebook (`paper_figures/embed/`),
and the scripts used to build everything (`scripts/`). The notebook-assembly scripts
(`restructure_notebook.py`, `add_receptive_field_section.py`, `make_english_notebook.py`) are
kept as a record of how the notebook was put together; they operate on an earlier
Chinese-language draft that is not part of this repository.

## What is *not* included, and how to get it

The preprocessed PyTorch Geometric datasets are far too large for git:

| File | Size |
|---|---|
| `database/datasets/train/multiscale_mesh_dataset.pkl` | 2.0 GB |
| `database/datasets/test/multiscale_mesh_dataset.pkl` | 0.50 GB |
| `database/datasets/train/dijkring_15.pkl` | 58 MB |
| `database/datasets/test/dijkring_15.pkl` | 0.58 GB |

Only two of the four are needed for sections 0–4.1 and 5:
`train/multiscale_mesh_dataset.pkl` + `test/multiscale_mesh_dataset.pkl` (2.5 GB together).
The two `dijkring_15.pkl` files (0.64 GB) are needed only for section 4.2. The
`mesh_dataset.pkl` pair that `scripts/build_real_pickles.py` also produces is **not used by
this notebook at all** — drop those two jobs from the script's job list to save 1.6 GB and
about a third of the build time.

Note that the **training** split is required even though the notebook never trains: the
feature-normalisation scalers are fitted on it, so a test-only download will not work.

To rebuild them:

1. Download `raw_datasets_mesh.zip` (2.75 GB) and `raw_datasets_dk15.zip` (0.92 GB) from
   Zenodo [10.5281/zenodo.13326595](https://doi.org/10.5281/zenodo.13326595) and unzip them
   into `database/raw_datasets_mesh/` and `database/raw_datasets_dk15/`.
2. Run `python scripts/build_real_pickles.py` (about 13 minutes; 3 s per simulation).

Budget the disk before starting: the two zips (3.7 GB) plus their unpacked contents
(~8.4 GB) plus the pickles (4.5 GB) peak at roughly 16 GB if nothing is deleted along the
way. Delete each zip right after unzipping it, and delete `database/raw_datasets_*/` once the
pickles exist, and the peak drops to about 9 GB. On a quota-limited home directory this is
usually the binding constraint, not the environment.

## Running the notebook

Two environment files pin the exact set the notebook was executed with, reduced to what it
actually needs. Pick by whether you have a GPU:

```bash
conda env create -f environment-cpu.yml -p ~/envs/mswegnn   # CPU, ~200 MB of torch
# or
conda env create -f environment.yml     -p ~/envs/mswegnn   # CUDA, ~2.5 GB of torch
conda activate ~/envs/mswegnn
jupyter lab test_pretrained_EN.ipynb
```

Two things about that file are worth knowing before you substitute your own:

- **`lightning` must be exactly 2.0.9.post0.** The upstream code calls
  `plmodule.load_from_checkpoint(...)` on an *instance*, which 2.1+ rejects with a `TypeError`.
- **`torch_scatter` / `torch_sparse` / `pyg_lib` are not needed.** This model runs on PyG
  2.4's pure-torch paths, which removes the part of a PyG install most likely to fail.
  `meshkernel`, `triangle`, `netCDF4`, `xarray`, `shapely` and `networkx` are needed only so
  that `database/graph_creation.py` imports; the notebook uses just two plotting functions
  from it.

### Runtime, and running without a GPU

A GPU is convenient, not required. Measured on an Intel Xeon Gold 6426Y, the same 20-simulation
rollout the notebook performs in section 4.1.1:

| Threads | Per simulation | 20 sims (§4.1.1) | 16 checkpoints × 20 sims (§4.1.5) |
|---|---|---|---|
| 4 | 16.6 s | 5.5 min | 88 min |
| 8 | 8.6 s | **2.9 min** | **46 min** |
| 64 | 13.9 s | 4.6 min | 74 min |
| 1×H100 | 0.19 s | 4 s | ~1 min |

Two things follow:

- **Everything except section 4.1.5 is comfortable on CPU.** The main rollout is minutes, not
  hours. Leave `RUN_ALL_CHECKPOINTS=True` if you are willing to wait ~45 min for the Pareto
  sweep; set it to `False` if you are not. Nothing else in the notebook needs changing, and no
  CUDA call in it is unguarded.
- **More threads is not better.** 8 threads beat 64 by 1.6×; the graphs are small enough that
  thread oversubscription costs more than the parallelism gains. If your container exposes many
  cores, run `torch.set_num_threads(8)` after the imports cell.

The dike-ring-15 section (§4.2) uses 10 simulations of roughly twice the node count, so expect
about 3 minutes at 8 threads (extrapolated, not measured).

### On the I-GUIDE Platform

None of the stock kernels on the [I-GUIDE Platform](https://platform.i-guide.io/) JupyterHub
will run this notebook as shipped. `geoai` is the closest (it has PyTorch) but lacks
`torch_geometric`, and the kernels live on a read-only CVMFS mount, so they cannot be patched
in place. Build your own:

```bash
bash setup_iguide_kernel.sh          # CPU build (default), inspects the stock kernels first
bash setup_iguide_kernel.sh --cuda   # only if a GPU is actually present
```

The kernel is registered as `mswegnn` because that is the kernelspec recorded in the
notebook. The script first prints what the stock kernels contain, so if one of them turns out
to be sufficient you can stop and just select it. It defaults to the CPU build, which is both
the right choice on a hub without GPUs and ~2.5 GB smaller — that difference matters against a
home-directory quota.

## Known issues

- **Flood extent reproduces to about 3 %**: CSI₀.₀₅ 0.803 here against the 0.830 recorded by
  the authors, CSI₀.₃ 0.658 against 0.687. The likeliest cause is that the pickles were
  rebuilt with meshkernel 3.0.0, so the coarse meshes may differ slightly. Not verified.
- **Depth error reproduces to about 2 %**, once compared like with like. The column named
  `test roll loss WD` in `overview_MSGNN.csv` holds an **MAE**, not an RMSE, despite its name
  and despite `config.yaml` setting `type_loss: RMSE`: over all 16 checkpoints our MAE differs
  from it by 0.0030 on average (ratio 1.00) while our RMSE differs by 0.0367 (ratio 1.60,
  which is exactly our own RMSE/MAE ratio). An earlier version of this README reported the
  depth error as not reproducing; that was our comparison error, not the authors'.
- **`lightning` must be pinned to 2.0.9.post0.** Version ≥2.1 raises a `TypeError` because
  `plmodule.load_from_checkpoint` is called on an instance. `environment.yml` pins it.
- **The SLURM scripts in `scripts/` were written for one specific cluster.** Absolute paths
  have been replaced by paths resolved from the script location and by `conda activate
  mswegnn`, but the partition names, account and resource requests still need editing.
  The notebook itself uses only relative paths and is portable as it stands.

## Attribution and licensing

- The **source code** is that of the upstream mSWE-GNN repository by Roberto Bentivoglio,
  redistributed under the **MIT licence**; the original `LICENSE` is preserved unchanged.
  The notebook, the analysis in section 2.2.1, and the scripts in `scripts/` are additions
  made in this fork.
- The **paper figures** in `paper_figures/` and embedded in the notebook are reproduced from
  Bentivoglio et al., NHESS 25, 335–351, 2025, under **CC BY 4.0**. Every figure carries a
  source line in the notebook cell that shows it.
- The **simulation data** is the authors' Zenodo record and is not redistributed here.

If you use the method, please cite the paper:

```bibtex
@article{bentivoglio2025mswegnn,
  author  = {Bentivoglio, Roberto and Isufi, Elvin and Jonkman, Sebastiaan Nicolas and Taormina, Riccardo},
  title   = {Multi-scale hydraulic graph neural networks for flood modelling},
  journal = {Natural Hazards and Earth System Sciences},
  volume  = {25},
  pages   = {335--351},
  year    = {2025},
  doi     = {10.5194/nhess-25-335-2025}
}
```
