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

To rebuild them:

1. Download `raw_datasets_mesh.zip` (2.75 GB) and `raw_datasets_dk15.zip` (0.92 GB) from
   Zenodo [10.5281/zenodo.13326595](https://doi.org/10.5281/zenodo.13326595) and unzip them
   into `database/raw_datasets_mesh/` and `database/raw_datasets_dk15/`.
2. Run `python scripts/build_real_pickles.py` (about 13 minutes; 3 s per simulation).

Note that all four pickles are needed even though the notebook never trains: the training
split is used to fit the feature-normalisation scalers.

## Running the notebook

```bash
conda create -n mswegnn python=3.10
conda activate mswegnn
pip install -r requirements.txt      # lightning must be 2.0.9.post0, see below
jupyter lab test_pretrained_EN.ipynb
```

The whole notebook takes about 20–25 minutes on one H100. Set `RUN_ALL_CHECKPOINTS=False`
in section 4.1.5 to skip the 16-checkpoint sweep, which is slow on CPU.

## Known issues

- **Water-depth RMSE does not reproduce exactly.** The flood-extent metrics match the
  authors' records closely (CSI₀.₀₅ 0.803 here versus 0.830 recorded), but the depth RMSE
  is 0.084 m against the recorded 0.050 m. The cause has not been identified; candidates
  are the pickles being rebuilt with meshkernel 3.0.0, mixed precision, and the
  `*_dataset2` dataset variants used in the authors' own notebook. Section 6 of the
  notebook records this in full.
- **`lightning` must be pinned to 2.0.9.post0.** Version ≥2.1 raises a `TypeError` because
  `plmodule.load_from_checkpoint` is called on an instance.
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
