"""Revise both notebooks for publication on the I-GUIDE Platform.

Applied to test_pretrained_EN.ipynb and test_pretrained_EN_fast.ipynb:

  1. drop section 5.1 (transfer to an unrelated in-house model) and the forward reference to it
  2. add section 0.0 "Before you run this": kernel, data, runtimes, and success criteria
  3. rewrite section 6's pitfalls: the I-GUIDE hub's, not the cluster the run happened on
  4. correct the environment line and the runtime claim in the title cell
  5. warn at 4.1.1 to check what each metric is before comparing
  6. shorten machine-specific absolute paths inside saved warning messages

Markdown only; no code cell and no saved result is altered.

    python plan/revise_for_iguide.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

DATA_STEPS = """```bash
wget -c -O database/raw_datasets_mesh.zip "https://zenodo.org/records/13326595/files/raw_datasets_mesh.zip?download=1"
wget -c -O database/raw_datasets_dk15.zip "https://zenodo.org/records/13326595/files/raw_datasets_dk15.zip?download=1"
python -c "import zipfile; [zipfile.ZipFile(f'database/raw_datasets_{n}.zip').extractall('database') for n in ('mesh','dk15')]"
python scripts/build_real_pickles.py
```"""


def prelude(kind):
    """Section 0.0, parameterised for the full and the subsampled notebook."""
    if kind == "full":
        runtimes = """| | CPU container | 1×H100 |
|---|---|---|
| §4.1.1, 20 simulations | ~13 min | 4 s |
| §4.1.5, 16 checkpoints | ~3.5 h | ~1 min |
| §4.2, dike ring 15 | ~13 min | ~10 s |
| **whole notebook** | **~4 h** | **~20 min** |

On CPU, set `RUN_ALL_CHECKPOINTS = False` in §4.1.5. It is the only cell worth skipping: the
Pareto figure below it is still drawn from the authors' recorded csv. Alternatively run
`test_pretrained_EN_fast.ipynb`, which subsamples the test sets and finishes in about
8 minutes."""
        expect = """| Quantity | Expect | If it differs |
|---|---|---|
| `parameters:` | exactly **811309** | wrong checkpoint, or `models.K` / `hid_features` edited |
| `test CSI_005` (§4.1.1) | **0.78 – 0.83** | below 0.7 means the pickles were built wrong |
| `MAE WD` (§4.1.1) | **≈ 0.051 m** | this is the paper's headline claim of 0.05 m |
| `test CSI_005` (§4.2) | **≈ 0.88** | the paper's 87.68 % after single-sample fine-tuning |"""
    else:
        runtimes = """| | CPU container | 1×H100 |
|---|---|---|
| §4.1.1, 3 simulations | ~2 min | 1.4 s |
| §4.1.5, skipped | — | — |
| §4.2, 2 dike-ring-15 simulations | ~3 min | ~3 s |
| **whole notebook** | **~8 min** | **~2 min** |"""
        expect = """| Quantity | Expect | If it differs |
|---|---|---|
| `parameters:` | exactly **811309** | wrong checkpoint, or `models.K` / `hid_features` edited |
| `test CSI_005` (§4.1.1) | **≈ 0.79** | below 0.7 means the pickles were built wrong |
| `test CSI_005` (§4.2) | **≈ 0.86** | — |

These are 3-simulation figures. The 20-simulation values, which are the ones comparable to the
paper, are in `test_pretrained_EN.ipynb`."""

    return f"""## 0.0 Before you run this

Every output below is already saved, so **this notebook can be read end to end without running
anything, without an environment and without the data.** The steps here are only needed to
re-execute it.

### 1. The kernel

Pick **`Python (mswegnn)`** from the kernel menu. If it is not listed, build it from a
Terminal:

```bash
bash setup_iguide_kernel.sh
```

That script uses `python -m venv` and pip **rather than conda**, on purpose. On this hub
`conda env create` is killed by the out-of-memory killer while parsing conda-forge repodata
and reports only `Killed`: the container is capped at 8 GiB and conda's classic solver exceeds
it. pip needs a fraction of that memory. Disk is not the constraint here — home is measured in
terabytes.

None of the roughly 26 kernels already installed on the hub will do. The closest, `geoai`, has
PyTorch 2.4.0 but no `torch_geometric`, and `lightning` must be **exactly 2.0.9.post0**,
because the upstream code calls `load_from_checkpoint` on an *instance*, which 2.1 and later
reject with a `TypeError`.

### 2. The data

The 3.1 GB of preprocessed pickles are not in the repository. Rebuild them from Zenodo:

{DATA_STEPS}

`unzip` is not installed on this hub, hence the Python one-liner. Zenodo sometimes answers
502 or 504 on the dike-ring-15 archive; `wget -c` resumes where it stopped. The build takes
about 13 minutes.

The **training** split is required even though nothing is trained here: the feature scalers are
fitted on it. A test-only download will not work.

### 3. What it will cost

Measured at roughly 40 s per synthetic simulation and 90 s per dike-ring-15 simulation on a
CPU container, against 0.19 s and 1.5 s on an H100:

{runtimes}

### 4. Did it work?

{expect}

Two structural checks that do not depend on hardware: the load should report
`train 60 / val 20 / test 20 simulations`, and test simulation 0 should have
**11 837 / 2 961 / 741 / 186** cells across the four scales. Those came out identical on two
unrelated clusters, so they are a good signal that the data pipeline rebuilt correctly."""


PITFALLS_OLD = """**Pitfalls when reproducing this on a cluster**
1. `lightning` must be 2.0.9.post0 (see requirements.txt); ≥2.1 raises a TypeError because `plmodule.load_from_checkpoint` is called on an instance.
2. The `WandbLogger` of Lightning ≥2 is lazy; this notebook calls `wandb.init(mode="disabled")` first and then `fix_dict_in_config`.
3. Zenodo's API and web pages returned 403 from this cluster, but `wget` on the direct file URLs works; `curl` is broken on this machine.
4. The polygon files in the data archive are lower-case `polygon_{i}.pol`; `create_mesh_dataset` was changed accordingly (`database/graph_creation.py:1612`).
5. For a full training run (200 epochs) use `sbatch plan/sbatch_gpu_real.sh` (setting `max_epochs` back to 200)."""

PITFALLS_NEW = """**Pitfalls, and which machine each one belongs to**

This was reproduced on two unrelated systems: an HPC cluster with H100 nodes, and the I-GUIDE
Platform's CPU JupyterHub. Items 1, 2 and 8 are properties of the code and will follow you
anywhere. Items 3 to 6 are the I-GUIDE hub. Item 7 was the cluster only, and is recorded
because it shows how much of "reproduction difficulty" is really local infrastructure.

1. **`lightning` must be exactly 2.0.9.post0** (see `requirements.txt`). Version 2.1 and later
   raise a `TypeError` because `plmodule.load_from_checkpoint` is called on an instance.
2. **Lightning ≥2's `WandbLogger` is lazy.** Calling `fix_dict_in_config` straight after
   constructing it raises `You must call wandb.init() before wandb.config.keys`. This notebook
   calls `wandb.init(mode="disabled")` first.
3. **The container is capped at 8 GiB.** `conda env create` gets OOM-killed while parsing
   conda-forge repodata and prints only `Killed`. Build the environment with `python -m venv`
   and pip instead. Disk is not the constraint: home is measured in terabytes.
4. **`unzip` is not installed.** Python's `zipfile` module substitutes.
5. **None of the ~26 preinstalled kernels is usable.** `geoai` is the closest — PyTorch 2.4.0,
   no `torch_geometric` — and none of them carries the required `lightning` pin.
6. **Zenodo intermittently answers 502/504** on the dike-ring-15 archive. `wget -c` resumes.
7. On the HPC cluster, Zenodo's API and web pages returned 403 while direct file URLs worked,
   and the system `curl` was broken. Neither problem appeared on I-GUIDE.
8. **The archive ships lower-case `polygon_{i}.pol`** while `create_mesh_dataset` opened
   `Polygon_{i}.pol`. One word changed at `database/graph_creation.py:1612` — the only edit
   made to upstream code in this whole reproduction; the original is kept alongside it.
9. For a full training run (200 epochs) use `sbatch scripts/sbatch_gpu_real.sh` with
   `max_epochs` set back to 200."""

ENV_OLD = """**Environment**: conda env `mswegnn` (torch 2.1.0, torch_geometric 2.4.0, lightning 2.0.9.post0). The whole notebook takes about 20–25 min on 1×H100; section 4.1.5 is slow on CPU, set `RUN_ALL_CHECKPOINTS=False` to skip it."""

ENV_NEW = """**Environment**: Python 3.10 or 3.11 with torch 2.1.0, torch_geometric 2.4.0 and lightning 2.0.9.post0 — `environment-cpu.yml` and `requirements-cpu.txt` pin the exact set, and **§0.0 below has the build, data and runtime steps for the I-GUIDE Platform**, including why conda cannot be used there."""

FORESHADOW = """#### 4.1.1 In the code: global metrics for `K4_F64` against the repo's `overview_MSGNN.csv`

> Before reading the table: **check what each column *is*, not what it is called.** The depth
> error in it was compared wrongly twice before the mismatch was found — `test roll loss WD`
> and `MAE WD` are different metrics, and the recorded csv column does not contain what its
> name and the config file both claim. Section 6 records how that went, because it is the most
> transferable thing in this notebook."""

PATHS = [
    ("/projects/bcrm/wz53/Aging_dam/mSWE-GNN-main/", ""),
    ("/u/wz53/miniconda3/envs/mswegnn/", "<env>/"),
    ("/home/jovyan/", "~/"),
]


def revise(path, kind):
    nb = nbf.read(path, as_version=4)
    log = []

    # 1. drop section 5.1 and the forward reference to it
    n0 = len(nb.cells)
    nb.cells = [c for c in nb.cells
                if not (c.cell_type == "markdown" and c.source.startswith("### 5.1"))]
    assert len(nb.cells) == n0 - 1, "section 5.1 not found"
    log.append("dropped section 5.1")

    ref = ("\n\nThis split matters again in §5.1: it decides which half of the method can be "
           "carried over to a static, single-pass model.")
    hit = [c for c in nb.cells if c.cell_type == "markdown" and ref in c.source]
    assert len(hit) == 1, "forward reference to 5.1 not found"
    hit[0].source = hit[0].source.replace(ref, "")
    log.append("removed the forward reference")

    # 2-5. markdown substitutions
    for old, new, label in [(ENV_OLD, ENV_NEW, "environment line"),
                            (PITFALLS_OLD, PITFALLS_NEW, "section 6 pitfalls"),
                            ("#### 4.1.1 In the code: global metrics for `K4_F64` against the "
                             "repo's `overview_MSGNN.csv`", FORESHADOW, "4.1.1 warning"),
                            ("## 6. Reproduction findings and cluster-specific pitfalls",
                             "## 6. Reproduction findings, and what the platform cost",
                             "section 6 title")]:
        hits = [c for c in nb.cells if c.cell_type == "markdown" and old in c.source]
        assert len(hits) == 1, f"{label}: matched {len(hits)}"
        hits[0].source = hits[0].source.replace(old, new)
        log.append(label)

    # section 0.0, right before the existing section 0
    idx = next(i for i, c in enumerate(nb.cells)
               if c.cell_type == "markdown" and c.source.startswith("## 0. Reproduction setup"))
    nb.cells.insert(idx, nbf.v4.new_markdown_cell(prelude(kind)))
    log.append("inserted section 0.0")

    # the contents table in the title cell
    row_old = "| 0 | — | Reproduction setup: environment, data, checkpoints |"
    row_new = ("| 0 | — | **How to run this**: kernel, data, runtimes, success criteria; then "
               "environment and checkpoint checks |")
    t = nb.cells[0]
    assert row_old in t.source
    t.source = t.source.replace(row_old, row_new)
    log.append("contents row")

    # 6. shorten machine paths inside saved warning text
    n = 0
    for c in nb.cells:
        for o in c.get("outputs", []):
            for key in ("text",):
                if isinstance(o.get(key), str):
                    for a, b in PATHS:
                        if a in o[key]:
                            o[key] = o[key].replace(a, b); n += 1
            d = o.get("data") or {}
            if isinstance(d.get("text/plain"), str):
                for a, b in PATHS:
                    if a in d["text/plain"]:
                        d["text/plain"] = d["text/plain"].replace(a, b); n += 1
    log.append(f"shortened paths in {n} output blocks")

    nbf.validate(nb)
    nbf.write(nb, path)
    outs = sum(len(c.get("outputs", [])) for c in nb.cells if c.cell_type == "code")
    print(f"{Path(path).name}: {len(nb.cells)} cells, outputs {outs} (unchanged)")
    for l in log:
        print(f"    - {l}")


for name, kind in [("test_pretrained_EN.ipynb", "full"),
                   ("test_pretrained_EN_fast.ipynb", "fast")]:
    revise(ROOT / name, kind)
