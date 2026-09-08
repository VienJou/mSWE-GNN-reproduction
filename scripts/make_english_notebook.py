"""Build test_pretrained_EN.ipynb: an English translation of test_pretrained.ipynb.

Only markdown cells are translated. Code cells, their saved outputs and the embedded
figure attachments are copied byte-for-byte, so the English notebook shows exactly the
same executed results as the Chinese one and needs no re-execution.

    python plan/make_english_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "test_pretrained.ipynb"
DST = ROOT / "test_pretrained_EN.ipynb"

CREDIT = ("*Figure source: Bentivoglio et al., NHESS 25, 335–351, 2025, "
          "[doi:10.5194/nhess-25-335-2025](https://doi.org/10.5194/nhess-25-335-2025), CC BY 4.0.*")

EN = {}

EN[0] = r"""# mSWE-GNN: paper walkthrough + pretrained-model reproduction (Multi-scale hydraulic graph neural networks for flood modelling)

**Paper**: Bentivoglio, Isufi, Jonkman & Taormina, *Multi-scale hydraulic graph neural networks for flood modelling*, Nat. Hazards Earth Syst. Sci. 25, 335–351, 2025, [doi:10.5194/nhess-25-335-2025](https://doi.org/10.5194/nhess-25-335-2025) (open access, CC BY 4.0).
**Code**: `mSWE-GNN-main/` (upstream repo v1.1, Nov 2024, git 529bb92). This notebook lives in the repo root and imports the authors' own `utils/`, `models/` and `training/` modules.
**Data**: Zenodo [10.5281/zenodo.13326595](https://doi.org/10.5281/zenodo.13326595), unpacked into `database/raw_datasets_mesh/` (100 D-HYDRO simulations) and `database/raw_datasets_dk15/` (11 dike-ring-15 simulations), then converted into `database/datasets/{train,test}/*.pkl`.

## What this notebook does
Two layers of content, interleaved and ordered **the way the paper is written**:
1. **Paper walkthrough** (markdown + the paper's own figures, taken from the open-access article page and stored in `paper_figures/`): the research problem and the gap, the method (multi-scale mesh, architecture, boundary conditions, loss), the experimental setup, the results, the discussion.
2. **Reproduction** (code cells, already executed on 1×H100 with outputs saved): training is **completely frozen**. We only load the 17 checkpoints shipped with the repo (`results/Pareto_front/models/K{2..5}_F{16,32,50,64}.h5`, `results/finetuned_dk15.h5`), roll them out on the real test sets and visualise the result. Subsections titled "**In the code / data**" are the reproduction; everything else is the paper.

| Notebook section | Paper | Content |
|---|---|---|
| 0 | — | Reproduction setup: environment, data, checkpoints |
| 1 | §1 Introduction | The problem, the four limitations of SWE-GNN, the contribution |
| 2 | §2 Methodology | Overall framework (Fig 1), multi-scale mesh and graph (Fig 2), encoder/processor/decoder, boundary conditions (Fig 3), rotation-invariant inputs, loss (Fig 13) |
| 3 | §3 Experimental setup | Synthetic dataset (Fig 4, Fig 6, Table 1), dike ring 15 (Fig 5), normalisation, training setup (Table D1), metric definitions |
| 4 | §4 Results | Comparison with SWE-GNN (Fig 8, Fig 7, Fig 12, Table A1) + reproduction; transfer to dike ring 15 (Fig 9–11, Table 2) + reproduction; ablation (Table 3) |
| 5 | §5–6 + Appendix C | Strengths, limitations, future work; speed-up from parallel inference (Fig 14) |
| 6 | — | Reproduction findings and cluster-specific pitfalls |

**Environment**: conda env `mswegnn` (torch 2.1.0, torch_geometric 2.4.0, lightning 2.0.9.post0). The whole notebook takes about 20–25 min on 1×H100; section 4.1.5 is slow on CPU, set `RUN_ALL_CHECKPOINTS=False` to skip it.
"""

EN[1] = "## 0. Reproduction setup: environment, data and checkpoints"

EN[3] = r"""### 0.1 Data and checkpoint check
`config.yaml` is the authors' reference configuration; its `saved_model` field points at the checkpoint to be tested.
`create_model_dataset` loads both the train and the test pickle (the training split is used **only** to fit the feature-normalisation scalers — no training happens here)."""

EN[5] = r"""### 0.2 Load the test dataset (20 synthetic simulations)
Each sample is a PyG `Data` object: `WD`/`VX`/`VY` have shape `[num_nodes, 97]` (96 h, one frame per hour; the config's `temporal_res=120 min` subsamples every other frame into 49 steps); `node_ptr` delimits the node ranges of the 4 scales; `BC` is the inflow hydrograph."""

EN[7] = r"""## 1. Introduction: the problem and the gap

**The problem.** Dike- and dam-breach flood mapping relies on numerical models that solve the 2-D shallow water equations (SWE), such as D-HYDRO / Delft3D FM. They are accurate, but one 96-hour simulation takes hours (12 h 20 min for the 20 synthetic cases and 47 h 15 min for the 10 dike-ring-15 cases, see Table A1), which rules out probabilistic, many-scenario assessment. Deep-learning surrogates can be orders of magnitude faster, but most of them have to be retrained for every new area.

**The starting point.** The authors' earlier **SWE-GNN** (a hydraulic-based graph neural network) treats the finite-volume mesh as a graph and lets a GNN propagate water between neighbouring cells. It transfers to unseen domains and can embed physical constraints.

**The gap (the four limitations listed in the abstract):**
1. **It cannot represent large differences in propagation speed** — information travels one hop per layer, so a fast flood wave needs many layers.
2. **Training becomes unstable with many layers** — SWE-GNN needs 10–18 of them (Table D1); deep message passing is hard to train and slow at inference.
3. **It cannot take time-varying boundary conditions**, such as a breach discharge hydrograph.
4. **It needs initial conditions from a numerical solver.**

**The proposal: mSWE-GNN** (multi-scale SWE-GNN)
- Build **several mesh resolutions** over the same domain (4 scales) and arrange them into a **U-shaped** multi-scale GNN: fine → coarse → fine. One hop on a coarse scale covers a long physical distance, so a few layers per scale (2–5) let information cross the domain, while the finest scale (which holds most of the nodes) does *less* work than before.
- Inject the time-varying discharge hydrograph through a **ghost cell** as a directed edge, and roll out autoregressively from a dry bed ($t=0$, no water), with no dependence on a numerical solver.
- Use only **rotation-invariant** inputs (area, elevation, roughness, water depth, **magnitude** of unit discharge, dual edge length).
- The decoder is a 1-D convolution along the time axis plus an MLP that outputs the next $h$ and $|q|$ directly, with a ReLU enforcing non-negativity.

**Headline results** (from the abstract): MAE ≈ 0.05 m for water depth and ≈ 0.003 m² s⁻¹ for unit discharge on the synthetic test set; speed-up > 700× on the real case study (dike ring 15 in the Netherlands), reaching CSI$_{0.05\,\mathrm{m}}$ = 87.68 % after fine-tuning on a **single** simulation."""

EN[8] = rf"""## 2. Methodology

### 2.0 Overall framework (paper Fig. 1)

![Fig 1](attachment:f01.jpg)

*Fig. 1 —* The model $\Phi(\cdot)$ takes (blue box) a fine mesh and its progressively coarsened versions $\mathcal{{M}}_{{1..M}}$, the **static inputs** $\mathbf{{X}}_s$ defined on them (DEM and so on), the **dynamic inputs** $\mathbf{{U}}^{{t-p:t}}$ (water depth and discharge at the previous $p+1$ times) and the **boundary condition** (the current slice of the breach hydrograph); it outputs (orange box) the hydraulic variables at the next time, $\hat{{\mathbf{{U}}}}^{{t+1}}$. Feeding the output back in as the next input rolls the prediction forward to $T$. The lower half is the **multi-scale module**: a GNN runs on the fine scale (purple arrows), the state is downsampled (green) to the coarser scale for another GNN, and after the coarsest scale it is upsampled step by step (yellow-green) and added to the matching downward features (red skip connections), before the decoder produces the output.

{CREDIT}"""

EN[9] = rf"""### 2.1 Multi-scale mesh and multi-scale graph (§2.1, paper Fig. 2)

**Mesh generation** (only the boundary polygon of the area is needed): MeshKernel first produces a coarse mesh; splitting every edge in two and connecting the new points gives the next finer level; the result is then orthogonalised (required by Delft3D's staggered grid) and over-elongated cells are removed, leaving a mixture of triangles and quadrilaterals. Repeating this yields $M$ scales (4 in this paper).

**Graph definition**: nodes are cell barycentres, edges connect cells that share a face (the dual graph). The rule for connecting scales is simple: **if a fine cell's centre falls inside a coarse cell, a directed inter-scale edge is added between the two**.

![Fig 2](attachment:f02.png)

*Fig. 2 —* (a) A three-scale multi-scale graph can be written as one block adjacency matrix: the diagonal blocks $\mathbf{{A}}^m \in \mathbb{{R}}^{{N^m\times N^m}}$ are each scale's own adjacency matrix, and the off-diagonal blocks $\mathbf{{P}}^{{m\to n}} \in \mathbb{{R}}^{{N^m\times N^n}}$ ($n = m\pm1$) are the **prolongation matrices** between neighbouring scales; non-adjacent scales are zero. (b) Example connection between a fine mesh $\mathbf{{A}}^1$ and a coarse mesh $\mathbf{{A}}^2$: $\mathbf{{P}}^{{2\to1}}$ links one coarse cell to all the fine cells it covers.

**Downsampling / upsampling** (Eq. 5–6):
- fine → coarse: **mean pooling**, a coarse node's features are the average of the fine nodes attached to it (no parameters);
- coarse → fine: a **learnable** operator, $\mathbf{{h}}^{{m}}_{{d,i}} \leftarrow \psi_{{m+1\to m}}(\cdot)\cdot \mathbf{{h}}^{{m+1}}_{{d}}$, where multiplying by the coarse node's dynamic features guarantees that **water only propagates down to the fine cells if the coarse cell holds water**.

{CREDIT}"""

EN[10] = r"""#### In the code / data: the four mesh scales of test sim 0
Each simulation's domain is a random polygon; meshkernel builds a triangular mesh and refines it three times. The four scales hold roughly 10 000 / 2 500 / 600 / 150 cells. The coarse scales are what let information cross the domain within a few GNN layers (the long-range action of shallow-water waves)."""

EN[12] = r"""### 2.2 Architecture: encoder → processor (U-shaped multi-scale GNN) → decoder (§2.2)

**Encoder (Eq. 2)** — three shared three-layer MLPs map the inputs into a latent space of dimension $G$:
- static node features $\mathbf{x}_{s,i} = (a_i, e_i, m_i, w_i)$: cell area, elevation, Manning roughness, (boundary) water level → $\phi_s$;
- dynamic node features $\mathbf{x}^t_{d,i} = (h^{t-p:t}_i, |q|^{t-p:t}_i)$: water depth and unit-discharge magnitude over the previous $p+1$ times ($p=2$; the code's `previous_t=3` is the same 3 frames) → $\phi_d$;
- edge features $\boldsymbol{\varepsilon}_{ij} = (l_{ij})$: dual edge length → $\phi_\varepsilon$.
Static and dynamic features on the coarse scales come from mean pooling the finest scale.

**Processor: the GNN layer (Eq. 3–4)**

$$\mathbf{s}^{(\ell+1)}_{ij} = \psi\big(\mathbf{h}_{s,i},\ \mathbf{h}_{s,j},\ \mathbf{h}^{(\ell)}_{d,i},\ \mathbf{h}^{(\ell)}_{d,j},\ \boldsymbol{\varepsilon}'_{ij}\odot(\mathbf{h}^{(\ell)}_{d,j}-\mathbf{h}^{(\ell)}_{d,i})\big),\qquad
\mathbf{h}^{(\ell+1)}_{d,i} = \mathbf{h}^{(\ell)}_{d,i} + \sum_{j\in\mathcal{N}_i}\mathbf{s}^{(\ell+1)}_{ij}\mathbf{W}^{(\ell+1)}$$

$\psi:\mathbb{R}^{5G}\to\mathbb{R}^{G}$ is an MLP and $\mathbf{W}\in\mathbb{R}^{G\times G}$ is learnable. The difference term $\mathbf{h}_{d,j}-\mathbf{h}_{d,i}$ models the **hydraulic gradient**: only a node that already holds water can push water to its neighbours. That is the physical constraint built into the model, and the reason it is called "hydraulic-based".

**U-shaped ordering**: $L$ GNN layers per scale (`K` in the code), downsampling fine → coarse and upsampling coarse → fine; after upsampling, the features are added to the downward branch at the same scale (**skip connection**, Eq. 7: $\mathbf{h}^m_d \leftarrow \mathbf{h}^{m\downarrow}_d + \mathbf{h}^{m\uparrow}_d$) before the next group of GNN layers.

**Decoder (Eq. 8)**

$$\hat{u}^{t+1}_i = \mathrm{ReLU}\big(\mathbf{U}^{t-p:t}_i\,\mathbf{w}_p + \varphi(\mathbf{h}_{d,i})\big)$$

The first term is a **1-D convolution** along the time axis ($\mathbf{w}_p\in\mathbb{R}^{p+1}$ learnable, effectively a linear extrapolation of the last few frames); the second is a three-layer MLP decoding the latent features. The ReLU keeps depths and discharges non-negative. Note that this predicts the **absolute value at the next step**, not an increment as SWE-GNN does (Table 3 ablates a residual decoder).

**The configuration selected in the paper**: $L=4$ layers per scale, $G=64$, 4 scales, about **811 000** parameters — that is `K4_F64.h5` in the repo, loaded in section 3.4 below. `Architecture.png` in the repo root is a high-resolution diagram of the same architecture."""

EN[13] = r"""### 2.2.1 Receptive field and compute cost: why "4 layers per scale" is enough

The sentence above — "the coarse scales let information cross the domain within a few GNN layers" — needs to be made quantitative, because it is only half right.

**Premise: the refinement rule fixes the ratios between scales.** Refinement splits every edge in two, so neighbouring scales differ by **a factor 2 in length and a factor 4 in cell count**. Measured on test sim 0 (printed by the cell above) and on dike ring 15 (§4.2); the hop distance is the median distance between neighbouring cell barycentres, `face_distance`, read from the un-normalised pickles:

| Scale | Cells, synthetic test sim 0 | Hop | Cells, dike ring 15 | Hop |
|---|---|---|---|---|
| $\mathcal{M}_1$ finest | 11 837 | 91 m | 22 881 | 98 m |
| $\mathcal{M}_2$ | 2 961 (÷4.00) | 185 m (×2.02) | 5 724 (÷4.00) | 198 m (×2.02) |
| $\mathcal{M}_3$ | 741 (÷4.00) | 371 m (×2.00) | 1 433 (÷3.99) | 396 m (×2.00) |
| $\mathcal{M}_4$ bottleneck | 186 (÷3.98) | 739 m (×1.99) | 359 (÷3.99) | 785 m (×1.98) |

The ratios are exactly 4 and 2. The `node_ptr [0, 22881, 28605, 30038, 30397]` printed in §4.2 says the same thing: $22\,881\times(1+\tfrac14+\tfrac1{16}+\tfrac1{64}) \approx 30\,400$, i.e. the "30 397 nodes" are the sum over the four scales, and the finest scale alone holds 22 881.

**The receptive field accumulates along the U path.** Each scale's GNN module performs $K=4$ hops (`models.K` in `config.yaml`; the code builds `self.K = [K]*4 + [K]*3`, i.e. 7 modules), and the hop distance doubles with each scale:

$$\text{RF} = K\,(h_1 + 2h_1 + 4h_1 + 8h_1 + 4h_1 + 2h_1 + h_1) = 4\times 22\,h_1 = 88\,h_1$$

With the measured hop distances that is **8.1 km** for the synthetic dataset and **8.7 km** for dike ring 15.

**Compare with a single scale.** Covering the same 88 hops on the finest mesh alone would take **88 GNN layers**. SWE-GNN uses 10–18 (Table D1), giving a receptive field of only 0.9–1.6 km — this is the quantitative form of the paper's gap #1 and gap #2.

**Where the cost is saved.** 7 modules × 4 hops = 28 layers, but only 8 of them (4 down + 4 up) run on the finest mesh; the rest run on meshes holding 1/4, 1/16 and 1/64 of the nodes. Weighted by node count:

$$4\times\Big(1+\tfrac14+\tfrac1{16}+\tfrac1{64}+\tfrac1{16}+\tfrac14+1\Big) = 10.6\ \text{finest-scale-layer equivalents}$$

**About 10.6 layers of compute buys an 88-layer receptive field**, roughly an 8× saving. That resolves the apparent contradiction in Fig. 7: mSWE-GNN has *more* parameters (811 k) yet runs *faster* (speed-ups up to 1200×). The backpropagation path also shrinks from 88 layers to 28, which is the mechanism behind the paper's "more stable training".

**One clarification: a single forward pass does not span the whole domain.** A single hop at the bottleneck is only 0.74 km and the whole U path reaches about 8 km, whereas the synthetic domain is about 13 km across and dike ring 15 is about 36 km along its long side. One forward pass therefore covers roughly 63 % and 24 % of the respective domains.

Domain-scale coupling comes from **autoregression** instead: over 48 time steps the accumulated receptive field is $48\times 8 \approx 380$ km, far beyond any real domain. Information travels like a wave, step by step, rather than becoming globally visible in one pass. So mSWE-GNN carries long-range dependence with **two** mechanisms:
- **multi-scale** matches the per-step propagation distance to the real flood wave speed (a flood advances on the order of kilometres within a 2 h step, which is what the 8 km single-step receptive field covers), instead of stacking dozens of layers to chase one time step;
- **autoregression** spreads the domain-scale coupling over the 48 steps.

This split matters again in §5.1: it decides which half of the method can be carried over to a static, single-pass model."""

EN[14] = rf"""### 2.3 Boundary conditions: ghost cells (§2.3, paper Fig. 3)

![Fig 3](attachment:f03.png)

*Fig. 3 —* (a) A **ghost cell** (red) is added next to the boundary cell that receives a boundary condition: it belongs to the computational graph but not to the physical domain, and acts as the interface to the outside world. (b) In the dual graph an inflow boundary is a directed edge **from the ghost cell into the domain cell**, an outflow the other way round; wall boundaries get no ghost cell.

The discharge hydrograph $Q(t)$ [m³ s⁻¹] is first divided by the length of the edge it crosses, turning it into a unit discharge [m² s⁻¹] exactly as in the numerical method, and is then fed step by step as the ghost cell's dynamic feature; a water-level boundary simply imposes the known value on the ghost cell. **Time-varying boundary conditions** therefore enter message passing through the graph structure itself, which SWE-GNN could not do.

{CREDIT}"""

EN[15] = r"""#### In the code / data: DEM, the inflow location matching the ghost cell, and the 20 test hydrographs
Left: the terrain (DEM) on the finest scale and the inflow location (red dot = the boundary cell that the ghost cell feeds). Right: the inflow hydrographs of the 20 test simulations."""

EN[17] = r"""### 2.4 Rotation-invariant inputs (§2.4)

The model's outputs ($h$, $|q|$) are scalars, so the authors deliberately **avoid every direction-dependent feature**: no $x/y$ components of the slope, no edge orientation, the dynamic input uses the **magnitude** of unit discharge rather than its vector components, and the only edge feature is the dual edge length. Rotating the whole input therefore leaves the output unchanged (rotation invariance), so the model does not have to learn rotational symmetry from the data and is more sample-efficient. The ablation (Table 3, "rotation-dependent inputs") shows that switching to direction-dependent inputs raises the test RMSE from 0.052 to 0.061 m."""

EN[18] = rf"""### 2.5 Loss function and curriculum learning (§2.5, Appendix B, paper Fig. 13)

**Multi-step forecasting loss (Eq. 9)**

$$\mathcal{{L}}_f = \frac{{1}}{{HO}}\sum_{{\tau=1}}^{{H}}\sum_{{o=1}}^{{O}}\gamma_o\,\big\|\hat{{u}}^{{t+\tau}}_o - u^{{t+\tau}}_o\big\|^2,\qquad \gamma_1 = 1\ (h),\ \gamma_2 = 7\ (|q|)$$

- $H$ is the rollout length used in training, increased by **curriculum learning** from 1 step up to $H=6$ steps (12 h), which teaches the model to correct its own accumulating error;
- the code's `only_where_water` evaluates the loss only on wet cells, so the large dry areas do not dominate the gradient.

**Mass-conservation term (Appendix B, Eq. B1)**: $\mathcal{{L}}_c = \sum_i a_i\,\Delta\hat h_i - Q\,\Delta t$, i.e. the change of water volume inside the domain should equal the inflow volume; the total loss is $\mathcal{{L}} = \mathcal{{L}}_f + \alpha\,\mathcal{{L}}_c$.

![Fig 13](attachment:f13.png)

*Fig. 13 —* Sampling $\alpha_m$ log-uniformly in $[10^{{-8}}, 5\times10^{{-5}}]$ produced no statistically significant improvement in validation loss or CSI$_{{0.05}}$ ($p$ = 0.42 / 0.48), so the final model uses $\alpha = 0$ (bold in Table D1). The reproduction code in section 5 redraws this figure from `results/mass_conservation.csv`.

{CREDIT}"""

EN[19] = rf"""## 3. Experimental setup

### 3.1 Synthetic dataset (§3.1, paper Fig. 4, Fig. 6, Table 1)

- **Numerical model**: Delft3D FM (D-HYDRO Suite 1D2D), **100** dike-breach flood simulations: 60 train / 20 validation / 20 test.
- **Domain**: randomly generated ellipse-like polygons; the **DEM** is Perlin noise plus a small slope in a random direction; 4 mesh scales; spatially uniform Manning roughness of $0.023\ \mathrm{{m^{{-1/3}}\,s}}$.
- **Boundary condition**: an inflow hydrograph imposed on one random boundary edge, shaped like a Weibull density (right-tailed, as breach hydrographs are), with peaks of 150–300 m³ s⁻¹.
- **Time**: 2 h resolution over 96 h, i.e. 48 steps.

![Fig 4](attachment:f04.jpg)

*Fig. 4 —* The finest mesh and DEM of one simulation in the synthetic dataset; the red circle on the left is the ghost cell (the inflow boundary). The elevation spans only about −3 to +1 m, i.e. nearly flat polder terrain.

**Table 1 — statistics of the training / validation / test sets (finest mesh, mean ± SD)**

| | Training | Validation | Testing |
|---|---|---|---|
| Elevation [m] | 0.07 ± 0.11 | 0.06 ± 0.10 | 0.07 ± 0.11 |
| Cells | 1621 ± 310 | 1608 ± 287 | 1650 ± 320 |
| Cell area [m²] | 1922 ± 1161 | 1957 ± 1210 | 1875 ± 1137 |
| Edge length [m] | 48.6 ± 16.2 | 49.2 ± 16.9 | 48.1 ± 15.8 |
| Total flood volume [10⁶ m³] | 0.71 ± 0.34 | 0.70 ± 0.31 | 0.72 ± 0.36 |

![Fig 6](attachment:f06.png)

*Fig. 6 —* Mean ± 1 SD (dashed lines are the extremes) of the hydrographs used for training (blue), synthetic testing (orange) and dike-ring-15 testing (green). Training and synthetic testing share the same distribution; the dike-ring-15 discharges are 3–5× larger and do not return to zero (about 9× the total flood volume of the synthetic set), which makes it a genuinely **out-of-distribution** transfer case.

{CREDIT}"""

EN[20] = "#### In the data: ground-truth water depth of test sim 0 over time"

EN[22] = rf"""### 3.2 Real case study: dike ring 15 (§3.2, paper Fig. 5)

![Fig 5](attachment:f05.jpg)

*Fig. 5 —* Dike ring 15 in the Netherlands (Krimpenerwaard–Lopikerwaard, between Rotterdam and Utrecht, EPSG:28992). It covers 31 400 ha, holds 201 500 inhabitants, and carries an expected flood damage of EUR 5.1 billion per event. Eleven roughly equidistant **breach locations** were selected along the ring: the red cross is the one used for fine-tuning (training + validation), the ten blue crosses are used for testing.

- Simplifications: all water bodies and every piece of infrastructure not already in the DEM were removed, and roughness is uniform.
- Hydrographs: a rising limb (the breach widening) followed by a slow decline that ends at non-zero discharge, peaking at 700–1000 m³ s⁻¹.
- Mesh: 22 880 cells on the finest scale (about 30 000 nodes per sample in the code, see section 4.2); depending on the breach location, the basin responds either like a bathtub or like a slope.

{CREDIT}"""

EN[23] = r"""### 3.3 Normalisation (§3.3)

Only **cell area** and **edge length** are z-scored, and the mean/variance are computed **separately for each scale** using the training set (coarse cells are orders of magnitude larger, so pooling all scales into one statistic would be meaningless). Every other variable (elevation, water depth, discharge) is left untouched. This is `config.scalers` in the code, fitted on the training set by `create_model_dataset` — in this notebook it is only used to transform the test data."""

EN[24] = r"""### 3.4 Training setup (§3.4, paper Table D1)

| Item | Value |
|---|---|
| Framework | PyTorch 2.0.1 + PyTorch Geometric 2.4; NVIDIA A100 80 GB |
| Optimiser / learning rate | Adam, initial 0.003, ×0.7 every 20 epochs |
| Epochs | 200 with early stopping; 16-bit mixed precision; gradient clipping at 1 |
| Curriculum learning | training rollout length raised gradually to $H=6$ |
| Training time | mSWE-GNN 2–15 h, SWE-GNN 5–30 h; dike-ring-15 fine-tuning about 20 min (5 min with fewer epochs) |

**Table D1 — hyperparameter ranges (bold = best validation loss)**

| DL model | Hyperparameter | Values' range (**best**) |
|---|---|---|
| All models | Initial learning rate | 0.003 |
| | Input previous time steps ($p$) | 2 |
| | Maximum training steps ahead ($H$) | 6 |
| | Optimizer | Adam |
| | Batch size | 12 |
| | $\alpha$ (mass-conservation weight) | **0**, $[10^{-8}, 5\times10^{-5}]$ |
| SWE-GNN | Embedding dimension ($G$) | 16, 32, 50, **64** |
| | Number of GNN layers ($L$) | 10, 12, 14, **16**, 18 |
| mSWE-GNN | Embedding dimension ($G$) | 16, 32, 50, **64** |
| | Number of GNN layers ($L$, per scale) | 2, 3, **4**, 5 |

Mapping onto the repo: `results/Pareto_front/models/K{L}_F{G}.h5` is exactly this 4 × 4 = 16-model grid for mSWE-GNN; in `config.yaml`, `models.K` is $L$ and `models.hid_features` is $G$."""

EN[25] = r"""#### In the code: load the pretrained `K4_F64` and roll it out (no training in this notebook)
- Build an `MSGNN` with the same hyperparameters as the checkpoint (hid 64, K 4, 4 scales, 811 309 parameters) and load the weights through `LightningTrainer.load_from_checkpoint`.
- Rollout: starting from a dry bed at t=0, the model uses only its own previous prediction plus the inflow boundary and advances 47 steps (94 h) autoregressively. This is how the paper evaluates, i.e. not teacher forcing."""

EN[27] = r"""### 3.5 Metrics (§3.5)

- **MAE / RMSE** (Eq. 10): the error over the whole rollout, per cell and per time step, computed separately for water depth $h$ and unit discharge $|q|$.
- **CSI** (critical success index, Eq. 11): $\mathrm{CSI}=\dfrac{TP}{TP+FP+FN}$, obtained by thresholding prediction and ground truth at 0.05 m ("flooded or not") and 0.3 m ("dangerous depth or not"); 1 means the flood extent matches exactly.
- **Speed-up**: numerical model runtime divided by deep-learning inference time; the deep-learning model is run **in parallel** over all test simulations (Appendix C discusses the effect of batch size).

**Conventions used in this notebook:**
- **test roll loss WD / V**: the **RMSE** of water depth and unit discharge over the whole rollout (the quantity `main.py` logs to wandb / csv). Note that `test_model.py` prints the **MAE** instead; the two are not comparable, so the table below reports both.
- **CSI$_{0.05}$ / CSI$_{0.3}$**: critical success index, treating depths > 0.05 m / 0.3 m as flooded, measuring the overlap between predicted and true flood extent (1 = perfect).
- **speed-up**: the ratio between the D-HYDRO runtime (`database/overview.csv`) and the model's rollout time.

The comparison rows come from `results/Pareto_front/overview_MSGNN.csv` shipped with the repo (the values the authors logged to wandb during training)."""

EN[28] = r"""## 4. Results

### 4.1 Synthetic test set: comparison with SWE-GNN (§4.1)

#### 4.1.1 In the code: global metrics for `K4_F64` against the repo's `overview_MSGNN.csv`"""

EN[30] = rf"""#### 4.1.2 How the metrics evolve over the rollout (paper Fig. 8)

![Fig 8](attachment:f08.png)

*Fig. 8 —* On the synthetic test set: (a) CSI$_{{0.05}}$ / CSI$_{{0.3}}$ and (b) the MAE of $h$ and $|q|$ as a function of lead time (shaded band = ±1 SD). CSI$_{{0.05}}$ stays above 0.78 across all 48 steps; CSI$_{{0.3}}$ is low over the first 20 h because the deep-water area is still small, so a few cells dominate the score. The depth MAE grows monotonically (autoregressive error accumulation), while the discharge MAE peaks around the flood peak and falls back as the inflow subsides. The cell below reproduces both panels with `SpatialAnalysis`.

{CREDIT}"""

EN[31] = r"""##### In the code: CSI and MAE over the rollout
Left: CSI as a function of lead time (shaded = standard deviation over the 20 simulations). Right: how the depth and discharge errors accumulate."""

EN[33] = r"""#### 4.1.3 In the code: per-simulation error ranking
`plot_loss_per_simulation` sorts the 20 test simulations by RMSE and shows each one's CSI alongside; it returns the sorted ids, which we use to pick the best and worst cases."""

EN[35] = r"""#### 4.1.4 In the code: visualising a single simulation
`PlotRollout.explore_rollout` is the authors' summary figure: DEM, ground-truth/predicted/error water-depth maps, discharge maps, and the error over time (`scale=0` is the finest mesh). We look at the lowest-error simulation first, then the highest."""

EN[38] = r"""##### Water depth and discharge at selected times (best simulation)
One row per time (8 h / 24 h / 48 h / 94 h): ground truth · prediction · difference."""

EN[40] = rf"""##### The matching figure in the paper: unit-discharge rollout on the synthetic test set (Appendix A, paper Fig. 12)

![Fig 12](attachment:f12.jpg)

*Fig. 12 —* $|q|$ for one synthetic test simulation on a logarithmic colour scale: ground truth (top), prediction (middle), difference (bottom). The model reproduces how the flood front advances into the domain and then recedes as the inflow weakens; the largest errors sit near the breach and along the path of the flood peak around 48 h, at a magnitude of 0.02 m² s⁻¹. The `compare_v_rollout` call above draws the same kind of figure.

{CREDIT}"""

EN[41] = r"""##### Flood arrival time (FAT) and mass conservation
FAT is the time at which each cell first exceeds 0.05 m of water depth, one of the quantities emergency planners care about most. The mass-conservation plot compares the cumulative inflow volume with the water volume inside the domain."""

EN[44] = r"""##### The multi-scale view
Predicted water depth on all four scales at the same instant (the model also maintains a water field on the coarse scales, which is what carries information across scales)."""

EN[46] = rf"""#### 4.1.5 Pareto front: speed versus accuracy (paper Fig. 7, Table A1)

![Fig 7](attachment:f07.png)

*Fig. 7 —* mSWE-GNN (circles) and SWE-GNN (crosses) in the plane of speed-up against (a) validation RMSE and (b) validation CSI$_{{0.05}}$; colour is the parameter count and the dashed red line is each model's Pareto front. The key points:
- the mSWE-GNN Pareto front **dominates** SWE-GNN's — 2–4× faster at equal accuracy, and about 0.04 m lower RMSE at equal speed;
- the fastest mSWE-GNN variant exceeds a **1200×** speed-up; it does not have fewer parameters than SWE-GNN, but it runs only **2–5 layers on the finest scale** (the one with most nodes and edges) whereas SWE-GNN needs 10–18, and that is where the compute difference comes from;
- fewer layers also make training more stable (gap 2). The reference model selected in the paper is $L=4, G=64$, a compromise between speed and accuracy.

**Table A1 — runtimes and speed-ups (the selected model)**

| Dataset | Numerical model | mSWE-GNN | Speed-up |
|---|---|---|---|
| Synthetic test set (20 sims) | 12 h 20 min | 0.61 ± 0.02 s | 728 ± 32 |
| Dike ring 15 (10 sims) | 47 h 15 min | 0.24 ± 0.01 s | 708 ± 24 |

Below we roll out each of the 16 checkpoints shipped with the repo and rebuild this Pareto figure (the paper uses the validation set, we use the test set).

{CREDIT}"""

EN[47] = r"""##### In the code: reproducing the Pareto front over 16 checkpoints (K ∈ {2,3,4,5} × F ∈ {16,32,50,64})
The paper draws a Pareto front for a family of models in the speed-versus-accuracy plane. Here each checkpoint is rolled out again and plotted next to the values recorded in the repo.
Set `RUN_ALL_CHECKPOINTS=False` to skip this (each model takes several minutes on CPU)."""

EN[51] = rf"""### 4.2 Transfer to the real case study: dike ring 15 (§4.2, paper Fig. 9–11, Table 2)

Applied directly to dike ring 15, a model trained only on the synthetic data reaches a CSI$_{{0.05}}$ of just 63 % (the hydrographs are out of distribution, see Fig. 6). After fine-tuning on a **single** simulation for about 20 min, the CSI$_{{0.05}}$ over the 10 test breaches rises to **87.68 %** and the depth MAE falls from 0.31 m to 0.12 m.

**Table 2 — effect of fine-tuning (10 test simulations, mean ± SD, finest mesh)**

| Fine-tuning | MAE $h$ [10⁻² m] ↓ | MAE $\lvert q\rvert$ [10⁻² m² s⁻¹] ↓ | CSI$_{{\tau=0.05\,\mathrm{{m}}}}$ [%] ↑ | CSI$_{{\tau=0.3\,\mathrm{{m}}}}$ [%] ↑ |
|---|---|---|---|---|
| No | 31.09 ± 5.42 | 3.37 ± 1.24 | 63.36 ± 19.54 | 46.06 ± 18.62 |
| Yes | **12.07 ± 4.19** | **2.08 ± 0.82** | **87.68 ± 10.3** | **81.82 ± 16.07** |

![Fig 9](attachment:f09.jpg)

*Fig. 9 —* Water-depth rollout for one dike-ring-15 test breach (red cross, top left): ground truth (top), prediction (middle), difference (bottom). The model captures the overall dynamics of the flood spreading north-east across the low polder; the errors are mainly a systematic slight overestimate after 48 h (purple) and a local underestimate near the breach.

![Fig 10](attachment:f10.png)

*Fig. 10 —* Flood arrival time for the same simulation (the moment each cell first exceeds 0.05 m). The predicted arrival-time field is nearly identical to the ground truth, with the differences concentrated at the tail of the flood front (±12–24 h). FAT is the quantity that matters most for evacuation planning.

![Fig 11](attachment:f11.png)

*Fig. 11 —* CSI$_{{0.05}}$ of the fine-tuned model at all 10 test breach locations: 0.82–0.94, showing that fine-tuning on a single breach generalises to the rest of the ring, with different inflow directions and basin responses.

{CREDIT}"""

EN[52] = r"""#### In the code: load `finetuned_dk15.h5` and roll it out on the 10 dike-ring-15 test simulations
`config_finetune.yaml` fine-tunes on 1 dike-ring-15 simulation and tests on the remaining 10 (22 880 fine-mesh cells, about 30 000 nodes per sample). Here we only load the authors' fine-tuned checkpoint and test it."""

EN[55] = r"""### 4.3 Ablation study (§4.3, paper Table 3)

| Configuration | Val RMSE [m] | Val CSI$_{0.05}$ | Test RMSE [m] | Test CSI$_{0.05}$ |
|---|---|---|---|---|
| **mSWE-GNN (full)** | **0.044** | **0.956** | **0.052** | **0.943** |
| w/o multi-scale module | 0.051 | 0.948 | 0.061 | 0.929 |
| learnable downsampling (instead of mean pooling) | 0.048 | 0.950 | 0.056 | 0.936 |
| w/o skip connections (Eq. 7) | 0.045 | 0.955 | 0.053 | 0.941 |
| residual decoder (instead of the 1-D CNN of Eq. 8) | 0.046 | 0.954 | 0.054 | 0.940 |
| rotation-dependent inputs | 0.051 | 0.947 | 0.061 | 0.928 |

*These numbers were transcribed from the article text; check them against the original table before quoting.* The conclusion: the **multi-scale module** and the **rotation-invariant inputs** contribute most (removing either raises the test RMSE by about 17 %); the skip connections and the decoder form matter less but point the same way. This maps back onto the gaps in section 1: multi-scale addresses the range of propagation speeds and the instability of deep stacks, rotation invariance improves sample efficiency."""

EN[56] = rf"""## 5. Discussion and conclusion (§5–6) + Appendix C

**Strengths (against the four gaps of section 1)**
1. The U-shaped multi-scale structure covers the domain with 2–5 layers per scale → it can represent fast and slow propagation, trains stably, and does little work on the finest scale → speed-ups of 700–1200× and a Pareto front that dominates SWE-GNN.
2. Ghost cells plus directed edges → a time-varying breach hydrograph can be used as an input.
3. Rolling out from a dry bed at $t=0$ → no numerical solver is needed for the initial condition.
4. Rotation-invariant inputs plus the graph structure → transfer to unseen meshes, topographies and boundary conditions, with a real case study needing only one simulation for fine-tuning.

**Limitations (stated by the authors)**: only dike-breach floods were evaluated (not river, coastal or pluvial floods); time-varying water-level boundaries were not tested; roughness is spatially uniform; the mesh must be generated top-down from a boundary polygon, so an existing fine mesh cannot be used directly; multiple simultaneous boundary conditions were not considered; the number of layers is the same on every scale; and there is no comparison against newer methods such as Fourier neural operators or neural fields.

**Future work**: time-varying breach-growth models; adding water bodies and linear elements (roads, secondary dikes); precipitation input and coupling with 1-D drainage for urban flooding; probabilistic multi-scenario frameworks and uncertainty quantification; PINN-style auto-differentiation losses; larger training sets to remove the need for fine-tuning; JIT/IPU acceleration.

### Appendix C: speed-up from parallel inference (paper Fig. 14)

![Fig 14](attachment:f14.png)

*Fig. 14 —* Speed-up of the 16 Pareto models on the synthetic test set as a function of batch size, i.e. how many simulations are inferred in parallel (both axes log). Running 20 simulations in parallel buys another ~4.5×; models with more layers (larger dots) and more parameters (darker) benefit less. Counting the runtime of the one numerical simulation needed for fine-tuning, the overall speed-up on the real case study is about 4–8×.

{CREDIT}"""

EN[57] = r"""#### In the code: the repo's recorded parallel-inference timings and mass-conservation experiment (read from csv, not re-run)
- `batch_prediction_times.csv`: inference time as a function of batch size (several simulations inferred in parallel).
- `mass_conservation.csv`: the effect of the mass-conservation loss weight (`trainer_options.conservation`) on accuracy."""

EN[59] = r"""### 5.1 Carrying this over to a static, single-pass raster model

Applying the conclusion of §2.2.1 to a static dam-break inundation model — a 19-channel raster U-Net that predicts a single maximum-inundation envelope rather than a time series — requires separating which of mSWE-GNN's two mechanisms can transfer.

**The two have different long-range structure**

| | mSWE-GNN | Static raster U-Net |
|---|---|---|
| Task | time evolution $h(t), q(t)$, 48 steps | static maximum-inundation envelope, one pass |
| Single-pass receptive field | 8.1 km (63 % of the domain) | capped by the 512 px crop, 30.7 km at 60 m/px |
| Domain scale | 13 km synthetic / 36 km dike ring 15 | canvases with a median long side of 159 km |
| Coupling beyond the local window | multi-scale (per step) + autoregression (across steps) | none: tiles are inferred independently, overlaps are only averaged |
| Source of long-range information | the architecture | hand-computed channels (flow distance, attenuated discharge, distance to the dam) |

**The multi-scale half transfers.** Adding a coarse global branch to the U-Net — encoding the whole canvas downsampled and concatenating it with the fine crop's bottleneck features — is exactly mSWE-GNN's multi-scale mechanism, and could in principle replace the hand-computed along-path channels.

**The autoregressive half does not.** A static envelope has no time axis, so there is no mechanism to spread the coupling over multiple steps; global context has to be supplied by the architecture in one shot. For a static task that may well be the right trade-off, but it should be stated explicitly, because "add a coarse branch and you have mSWE-GNN" would be wrong.

**A note on citing the numbers: do not compare against the 700× speed-up directly.** That ratio is measured against one 48-step Delft3D run. A static surrogate trained on published inundation maps replaces an entire modelling study rather than a single solver run, which is a different context. The same caution applies to CSI: it is the same quantity as binary IoU, but mSWE-GNN's 0.88 is a single dike ring, after fine-tuning, and scored against the same solver that generated the labels."""

EN[60] = r"""## 6. Reproduction findings and cluster-specific pitfalls
**Findings**
- Training can be skipped entirely: the 17 checkpoints shipped with the repo are compatible with the current code and data, and `test_model.py` (or this notebook) is enough to test them.
- `K4_F64` on the real test set: CSI$_{0.05}$ ≈ 0.80 and CSI$_{0.3}$ ≈ 0.66 (the repo records 0.83 / 0.69, within 0.03). But the water-depth **RMSE ≈ 0.084 m**, well above the 0.050 m recorded in the repo (the MAE of 0.051 m happens to be numerically close, but it is a different quantity). Flood-extent metrics reproduce well; pointwise depth error does not, and the cause is not yet identified.
- Possible sources of the difference, none of them verified: the pickles were rebuilt with meshkernel 3.0.0 (the coarse meshes may differ slightly from the authors'), AMP, and the `*_dataset2` dataset variants used in the authors' own notebook.

**Pitfalls when reproducing this on a cluster**
1. `lightning` must be 2.0.9.post0 (see requirements.txt); ≥2.1 raises a TypeError because `plmodule.load_from_checkpoint` is called on an instance.
2. The `WandbLogger` of Lightning ≥2 is lazy; this notebook calls `wandb.init(mode="disabled")` first and then `fix_dict_in_config`.
3. Zenodo's API and web pages returned 403 from this cluster, but `wget` on the direct file URLs works; `curl` is broken on this machine.
4. The polygon files in the data archive are lower-case `polygon_{i}.pol`; `create_mesh_dataset` was changed accordingly (`database/graph_creation.py:1612`).
5. For a full training run (200 epochs) use `sbatch plan/sbatch_gpu_real.sh` (setting `max_epochs` back to 200)."""

nb = nbf.read(SRC, as_version=4)
missing = [i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and i not in EN]
assert not missing, f"untranslated markdown cells: {missing}"

n_att = 0
for i, c in enumerate(nb.cells):
    if c.cell_type != "markdown":
        continue
    c.source = EN[i]
    n_att += len(getattr(c, "attachments", {}) or {})

nbf.validate(nb)
nbf.write(nb, DST)
n_code = sum(c.cell_type == "code" for c in nb.cells)
n_out = sum(len(c.outputs) for c in nb.cells if c.cell_type == "code")
print(f"wrote {DST}: {len(nb.cells)} cells ({n_code} code, {len(EN)} markdown translated), "
      f"{n_att} figure attachments kept, {n_out} outputs kept")
