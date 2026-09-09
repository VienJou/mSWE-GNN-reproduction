"""Build test_pretrained_EN_fast.ipynb: a fast, executable variant of the walkthrough.

The full notebook takes about 4 hours on a CPU-only JupyterHub container. This variant
subsamples the test sets and skips the 16-checkpoint sweep, bringing it to roughly 8
minutes, so the workflow can be demonstrated live.

Every edit keeps the original line commented out immediately above the replacement, so the
difference from the full notebook is visible in place.

All code-cell outputs are cleared: this file is meant to be executed, and stale outputs from
a 20-simulation run would contradict the 3-simulation code.

    python plan/make_fast_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "test_pretrained_EN.ipynb"
DST = ROOT / "test_pretrained_EN_fast.ipynb"

N_TEST, N_DK15 = 3, 2

nb = nbf.read(SRC, as_version=4)


def edit(cell_pred, old, new, label):
    """Replace `old` with `new` in the single code cell matching cell_pred."""
    hits = [c for c in nb.cells if c.cell_type == "code" and cell_pred(c.source)]
    assert len(hits) == 1, f"{label}: matched {len(hits)} cells"
    c = hits[0]
    assert c.source.count(old) == 1, f"{label}: anchor found {c.source.count(old)} times"
    c.source = c.source.replace(old, new)


# ---- 1. subsample the synthetic test set, right after it is loaded ------------------
edit(lambda s: "del train_dataset, val_dataset" in s,
     "del train_dataset, val_dataset   # not needed: no training in this notebook",
     f"""del train_dataset, val_dataset   # not needed: no training in this notebook

# ---------------- FAST VARIANT ----------------------------------------------------
# The full notebook evaluates all 20 test simulations, which takes ~13 min on a CPU-only
# container. Keeping the first {N_TEST} cuts every rollout below proportionally.
# load_dataset(seed=0) does not shuffle, so these are seeds 81..{80 + N_TEST}, and
# get_numerical_times(..., len(test_dataset)) slices the matching rows of overview.csv,
# so the speed-up figures stay correctly paired.
N_TEST = {N_TEST}
test_dataset = test_dataset[:N_TEST]
print(f"FAST VARIANT: keeping {{len(test_dataset)}} of 20 test simulations")
# ----------------------------------------------------------------------------------""",
     "subsample test_dataset")

# ---- 2. skip the 16-checkpoint sweep ------------------------------------------------
edit(lambda s: s.startswith("RUN_ALL_CHECKPOINTS"),
     "RUN_ALL_CHECKPOINTS = True",
     """# RUN_ALL_CHECKPOINTS = True    # full notebook: 16 checkpoints x 20 sims, ~3.5 h on CPU
# ---------------- FAST VARIANT ----------------------------------------------------
# Skipped. df_pf becomes None, and both cells below are guarded by `if df_pf is not None`,
# so the Pareto figure is still drawn from the authors' recorded overview_MSGNN.csv.
RUN_ALL_CHECKPOINTS = False
# ----------------------------------------------------------------------------------""",
     "skip checkpoint sweep")

# ---- 3. subsample dike ring 15 ------------------------------------------------------
edit(lambda s: "run_rollout(plm_ft" in s,
     'print(len(test_dk15), "dijkring-15 test simulations;"',
     f"""# ---------------- FAST VARIANT ----------------------------------------------------
# These meshes carry ~30 000 nodes each, about twice the synthetic ones, so each rollout
# costs roughly twice as much. {N_DK15} of the 10 breach locations is enough to show the
# transfer case; the per-location CSI figure below simply has fewer points.
N_DK15 = {N_DK15}
test_dk15 = test_dk15[:N_DK15]
# ----------------------------------------------------------------------------------
print(len(test_dk15), "dijkring-15 test simulations;\"""",
     "subsample test_dk15")

edit(lambda s: "run_rollout(plm_ft" in s,
     "pred_ft, t_ft = run_rollout(plm_ft, test_dk15, batch_size=5)",
     """# pred_ft, t_ft = run_rollout(plm_ft, test_dk15, batch_size=5)   # full notebook
pred_ft, t_ft = run_rollout(plm_ft, test_dk15, batch_size=2)     # FAST VARIANT""",
     "dk15 batch size")

# ---- 4. retitle, and add the notice at the top --------------------------------------
title = nb.cells[0]
assert title.cell_type == "markdown" and title.source.startswith("# mSWE-GNN")
title.source = title.source.replace(
    "# mSWE-GNN: paper walkthrough + pretrained-model reproduction",
    "# mSWE-GNN: paper walkthrough + pretrained-model reproduction — FAST VARIANT", 1)

NOTICE = f"""> ## Fast variant — read this first
>
> This is the **executable** version of the walkthrough, trimmed so it finishes in roughly
> **8 minutes on a CPU-only JupyterHub container** instead of about 4 hours. Three changes,
> each marked `FAST VARIANT` in the code with the original line left commented above it:
>
> | Change | Full notebook | Here | Saves |
> |---|---|---|---|
> | Synthetic test simulations | 20 | **{N_TEST}** | ~11 min |
> | 16-checkpoint Pareto sweep | run | **skipped**, plotted from the authors' csv | ~3.5 h |
> | Dike-ring-15 test simulations | 10 | **{N_DK15}** | ~11 min |
>
> **The metrics printed here are computed over {N_TEST} simulations, not 20, so they will not
> match the paper or the full notebook.** They are for checking that the workflow runs, not
> for quoting. For the reproduction numbers — CSI₀.₀₅ 0.803 against the authors' recorded
> 0.830, and the depth RMSE that does not reproduce — see `test_pretrained_EN.ipynb`, which
> carries the saved outputs of a full 20-simulation run on an H100 and needs no execution to
> read.
>
> Nothing else differs: the same checkpoints, the same data pipeline, the same autoregressive
> rollout, the same metric definitions and the same figures.
"""
nb.cells.insert(1, nbf.v4.new_markdown_cell(NOTICE))

# ---- 5. clear every output: this file is meant to be run ----------------------------
cleared = 0
for c in nb.cells:
    if c.cell_type == "code":
        cleared += len(c.get("outputs", []))
        c.outputs, c.execution_count = [], None

nbf.validate(nb)
nbf.write(nb, DST)
print(f"wrote {DST}")
print(f"  {len(nb.cells)} cells, {sum(c.cell_type == 'code' for c in nb.cells)} code, "
      f"{cleared} outputs cleared, {DST.stat().st_size / 1e6:.1f} MB")
