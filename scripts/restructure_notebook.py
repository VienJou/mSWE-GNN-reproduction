"""Reorganise test_pretrained.ipynb along the NHESS-2025 paper structure and embed the
paper figures (paper_figures/embed/*.{png,jpg}) as self-contained markdown attachments.

- No code cell is modified or re-executed; outputs are kept.
- Code cells keep their original relative order except cell 36 (csv-only, independent),
  which moves to the appendix section.

    python plan/restructure_notebook.py
"""
import base64
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "plan/test_pretrained_backup_v1_before_paper_figs.ipynb"
DST = ROOT / "test_pretrained.ipynb"
EMB = ROOT / "paper_figures/embed"

nb = nbf.read(SRC, as_version=4)
E = nb.cells                                     # original cells, indexed as in the survey
assert len(E) == 41, len(E)

ATTR = ("*图片来源：Bentivoglio et al., NHESS 25, 335–351, 2025, "
        "[doi:10.5194/nhess-25-335-2025](https://doi.org/10.5194/nhess-25-335-2025)，CC BY 4.0。*")


def fig_md(text, figs):
    """Markdown cell whose images are embedded as attachments (self-contained notebook)."""
    cell = nbf.v4.new_markdown_cell(text)
    cell.attachments = {}
    for f in figs:
        p = next(EMB.glob(f"{f}.*"))
        mime = "image/jpeg" if p.suffix == ".jpg" else "image/png"
        cell.attachments[p.name] = {mime: base64.b64encode(p.read_bytes()).decode()}
    return cell


def att(f):
    p = next(EMB.glob(f"{f}.*")); return f"attachment:{p.name}"


def rehead(cell, new_first_line):
    lines = cell.source.split("\n"); lines[0] = new_first_line
    cell.source = "\n".join(lines); return cell


md = nbf.v4.new_markdown_cell
new = []

# =============================================================================== 0 title
new.append(md(rf"""# mSWE-GNN：论文导读 + 预训练模型复现（Multi-scale hydraulic graph neural networks for flood modelling）

**论文**: Bentivoglio, Isufi, Jonkman & Taormina, *Multi-scale hydraulic graph neural networks for flood modelling*, Nat. Hazards Earth Syst. Sci. 25, 335–351, 2025, [doi:10.5194/nhess-25-335-2025](https://doi.org/10.5194/nhess-25-335-2025)（开放获取，CC BY 4.0）。
**代码**: `mSWE-GNN-main/`（repo v1.1, Nov 2024，git 529bb92）— 本 notebook 位于 repo 根目录，直接复用作者的 `utils/`、`models/`、`training/` 模块。
**数据**: Zenodo [10.5281/zenodo.13326595](https://doi.org/10.5281/zenodo.13326595)，已解压到 `database/raw_datasets_mesh/`（100 组 D-HYDRO 模拟）与 `database/raw_datasets_dk15/`（11 组 dijkring 15 模拟），并转换为 `database/datasets/{{train,test}}/*.pkl`。

## 这个 notebook 做什么
两层内容按 **论文的行文结构** 交织组织：
1. **论文导读**（markdown + 论文原图，图片来自开放获取的文章页面，存于 `paper_figures/`）：研究问题与 gap、方法（多尺度网格、网络结构、边界条件、损失）、实验设置、结果、讨论。
2. **复现**（代码 cell，输出已在 1×H100 上执行并保存）：训练阶段 **完全冻结**，只加载 repo 自带的 17 个 checkpoint（`results/Pareto_front/models/K{{2..5}}_F{{16,32,50,64}}.h5`, `results/finetuned_dk15.h5`），在真实测试集上做 rollout 并可视化。带"**在代码/数据中**"标题的小节是复现代码，其余是论文内容。

| notebook 章节 | 对应论文 | 内容 |
|---|---|---|
| 0 | — | 复现准备：环境、数据、checkpoint |
| 1 | §1 Introduction | 研究问题、已有 SWE-GNN 的四个局限、本文贡献 |
| 2 | §2 Methodology | 总体框架（Fig 1）、多尺度网格与图（Fig 2）、Encoder/Processor/Decoder、边界条件（Fig 3）、旋转不变输入、损失函数（Fig 13） |
| 3 | §3 Experimental setup | 合成数据集（Fig 4, Fig 6, Table 1）、dike ring 15（Fig 5）、归一化、训练设置（Table D1）、指标定义 |
| 4 | §4 Results | 与 SWE-GNN 的比较（Fig 8, Fig 7, Fig 12, Table A1）+ 复现；迁移到 dike ring 15（Fig 9–11, Table 2）+ 复现；消融（Table 3） |
| 5 | §5–6 + Appendix C | 优势、局限、未来工作；并行推理加速（Fig 14） |
| 6 | — | 复现结论与集群上的注意事项 |

**运行环境**: conda env `mswegnn`（torch 2.1.0, torch_geometric 2.4.0, lightning 2.0.9.post0）。整本 notebook 在 1×H100 上约 20–25 分钟；CPU 上第 4.1.5 节会很慢，可把 `RUN_ALL_CHECKPOINTS=False`。
"""))

# =============================================================================== 0 setup (E1..E6)
new.append(rehead(E[1], "## 0. 复现准备：环境、数据与 checkpoint"))
new.append(E[2])
new.append(rehead(E[3], "### 0.1 数据与 checkpoint 检查"))
new.append(E[4])
new.append(md("### 0.2 加载 test 数据集（20 组合成模拟）\n" + E[5].source.split("\n", 2)[2]))
new.append(E[6])

# =============================================================================== 1 Introduction
new.append(md(r"""## 1. Introduction：研究问题与 gap

**问题**：溃堤/溃坝洪水制图依赖求解二维浅水方程（SWE）的数值模型（如 D-HYDRO / Delft3D FM）。它们精度高，但一场 96 h 的模拟要以小时计（本文合成算例 12 h 20 min / 20 场，dike ring 15 为 47 h 15 min / 10 场，见 Table A1），无法支撑概率式、多情景的快速评估。深度学习代理模型可以把速度提高几个数量级，但多数模型换一个区域就要重新训练。

**已有方案**：作者此前提出的 **SWE-GNN**（hydraulic-based graph neural network）把有限体积网格当作图，用 GNN 模拟水在相邻单元间的传播，可以迁移到未见过的区域，并能嵌入物理约束。

**gap（摘要中列出的四个局限）**：
1. **无法表达传播速度的剧烈差异**——信息每层只走一跳，快速洪波需要很多层；
2. **层数多时训练不稳定**——SWE-GNN 需要 10–18 层才够用（Table D1），深层消息传递训练困难、推理也慢；
3. **不能接受随时间变化的边界条件**（如溃口流量过程线 hydrograph）；
4. **需要数值求解器提供初始条件**。

**本文方案：mSWE-GNN**（multi-scale SWE-GNN）
- 在同一域上建 **多个分辨率的网格**（4 个尺度），做成 **U 形** 的多尺度 GNN：细→粗→细。粗尺度上一跳覆盖很长的物理距离，所以少量层（每尺度 2–5 层）即可让信息横跨整个计算域，同时最细尺度（节点最多）的计算量反而更少。
- 用 **ghost cell** 把时变的流量 hydrograph 作为有向边注入图，从干河床（$t=0$ 无水）开始自回归预测，完全不依赖数值求解器。
- 输入全部是 **旋转不变** 量（面积、高程、糙率、水深、单宽流量 **模**、对偶边长）。
- 解码器用沿时间轴的 1D 卷积 + MLP 直接输出下一步的 $h$ 与 $|q|$，ReLU 保证非负。

**结果概览**（论文摘要）：合成测试集水深 MAE ≈ 0.05 m、单宽流量 MAE ≈ 0.003 m² s⁻¹；真实案例（荷兰 dike ring 15）加速 > 700 倍，只用 **1 场** 模拟微调后 CSI$_{0.05\,\mathrm{m}}$ = 87.68 %。
"""))

# =============================================================================== 2 Methodology
new.append(fig_md(rf"""## 2. Methodology

### 2.0 总体框架（论文 Fig. 1）

![Fig 1]({att('f01')})

*Fig. 1 —* 模型 $\Phi(\cdot)$ 的输入（蓝框）是一个细网格及其逐级粗化的版本 $\mathcal{{M}}_{{1..M}}$、定义在网格上的 **静态输入** $\mathbf{{X}}_s$（DEM 等）和 **动态输入** $\mathbf{{U}}^{{t-p:t}}$（前 $p+1$ 个时刻的水深/流量）以及 **边界条件**（溃口 hydrograph 的当前时段）；输出（橙框）是下一时刻的水力变量 $\hat{{\mathbf{{U}}}}^{{t+1}}$。把输出回填为下一步的输入，自回归地滚动到 $T$（rollout）。下半部分是 **multi-scale module**：细尺度先做 GNN（紫箭头），下采样（绿）到粗尺度再做 GNN，最粗尺度处理后逐级上采样（黄绿）并与同尺度的下行特征相加（红色 skip connection），最后由 decoder 输出。

{ATTR}
""", ["f01"]))

new.append(fig_md(rf"""### 2.1 多尺度网格与多尺度图（§2.1，论文 Fig. 2）

**网格生成**（只需要区域的边界多边形）：用 MeshKernel 先生成粗网格；把每条边二分、连接新点得到细一级网格；再做正交化（Delft3D 交错网格要求）并删除过于狭长的单元，得到三角形+四边形混合网格。重复多次即得到 $M$ 个尺度（本文 4 个）。

**图的定义**：节点 = 单元重心，边 = 相邻单元的公共边（对偶图）。跨尺度的连接规则很简单：**细网格单元的中心落在哪个粗单元内，就在两者之间加一条有向的 inter-scale edge**。

![Fig 2]({att('f02')})

*Fig. 2 —* (a) 三个尺度的多尺度图可以写成一个分块邻接矩阵：对角块 $\mathbf{{A}}^m \in \mathbb{{R}}^{{N^m\times N^m}}$ 是各尺度自己的邻接矩阵，次对角块 $\mathbf{{P}}^{{m\to n}} \in \mathbb{{R}}^{{N^m\times N^n}}$（$n = m\pm1$）是相邻尺度之间的 **prolongation matrix**；不相邻的尺度之间为 0。(b) 细网格 $\mathbf{{A}}^1$ 与粗网格 $\mathbf{{A}}^2$ 的连接示例：$\mathbf{{P}}^{{2\to1}}$ 把一个粗单元连到它覆盖的所有细单元。

**下采样 / 上采样**（Eq. 5–6）：
- 细→粗：**mean pooling**，粗节点的特征 = 它所连的细节点特征的平均（无参数）；
- 粗→细：**可学习** 的算子，$\mathbf{{h}}^{{m}}_{{d,i}} \leftarrow \psi_{{m+1\to m}}(\cdot)\cdot \mathbf{{h}}^{{m+1}}_{{d}}$，其中乘以粗节点的动态特征保证 **只有粗单元有水时才会向细单元传播**。

{ATTR}
""", ["f02"]))
new.append(rehead(E[7], "#### 在代码/数据中：test sim 0 的四尺度网格"))
new.append(E[8])

new.append(md(r"""### 2.2 网络结构：Encoder → Processor（U 形多尺度 GNN）→ Decoder（§2.2）

**Encoder（Eq. 2）**——三个共享的三层 MLP，把输入映射到维度 $G$ 的隐空间：
- 静态节点特征 $\mathbf{x}_{s,i} = (a_i, e_i, m_i, w_i)$：单元面积、高程、Manning 糙率、（边界）水位 → $\phi_s$；
- 动态节点特征 $\mathbf{x}^t_{d,i} = (h^{t-p:t}_i, |q|^{t-p:t}_i)$：前 $p+1$ 个时刻的水深与单宽流量模（$p=2$；代码里 `previous_t=3` 即 3 帧）→ $\phi_d$；
- 边特征 $\boldsymbol{\varepsilon}_{ij} = (l_{ij})$：对偶边长 → $\phi_\varepsilon$。
粗尺度上的静态/动态特征由最细尺度 mean pooling 得到。

**Processor：GNN 层（Eq. 3–4）**

$$\mathbf{s}^{(\ell+1)}_{ij} = \psi\big(\mathbf{h}_{s,i},\ \mathbf{h}_{s,j},\ \mathbf{h}^{(\ell)}_{d,i},\ \mathbf{h}^{(\ell)}_{d,j},\ \boldsymbol{\varepsilon}'_{ij}\odot(\mathbf{h}^{(\ell)}_{d,j}-\mathbf{h}^{(\ell)}_{d,i})\big),\qquad
\mathbf{h}^{(\ell+1)}_{d,i} = \mathbf{h}^{(\ell)}_{d,i} + \sum_{j\in\mathcal{N}_i}\mathbf{s}^{(\ell+1)}_{ij}\mathbf{W}^{(\ell+1)}$$

$\psi:\mathbb{R}^{5G}\to\mathbb{R}^{G}$ 是 MLP，$\mathbf{W}\in\mathbb{R}^{G\times G}$ 可学习。差分项 $\mathbf{h}_{d,j}-\mathbf{h}_{d,i}$ 模拟 **水力梯度**：只有已经有水的节点才能把水"推"给邻居，这是模型内建的物理约束，也是 "hydraulic-based" 的含义。

**U 形多尺度顺序**：在每个尺度上做 $L$ 层 GNN（代码里的 `K`），细→粗下采样、粗→细上采样；上采样后与下行分支同尺度的特征相加（**skip connection**, Eq. 7：$\mathbf{h}^m_d \leftarrow \mathbf{h}^{m\downarrow}_d + \mathbf{h}^{m\uparrow}_d$）再进入下一组 GNN 层。

**Decoder（Eq. 8）**

$$\hat{u}^{t+1}_i = \mathrm{ReLU}\big(\mathbf{U}^{t-p:t}_i\,\mathbf{w}_p + \varphi(\mathbf{h}_{d,i})\big)$$

第一项是沿时间轴的 **1D 卷积**（$\mathbf{w}_p\in\mathbb{R}^{p+1}$ 可学习，相当于对最近几帧做线性外推），第二项是三层 MLP 对隐特征的解码；ReLU 保证水深/流量非负。注意它 **直接输出下一时刻的值**，不是 SWE-GNN 那样只输出增量（消融 Table 3 比较了 residual decoder）。

**论文选定的配置**：每尺度 $L=4$ 层、$G=64$、4 个尺度，约 **811 000** 参数——即 repo 里的 `K4_F64.h5`（下面 3.4 节加载）。repo 根目录的 `Architecture.png` 是同一结构的高分辨率示意图。
"""))

new.append(fig_md(rf"""### 2.3 边界条件：ghost cell（§2.3，论文 Fig. 3）

![Fig 3]({att('f03')})

*Fig. 3 —* (a) 在接受边界条件的边界单元旁边加一个 **ghost cell**（红色）：它属于计算图但不属于物理域，是外部条件的"接口"。(b) 在对偶图中，入流边界是一条 **从 ghost cell 指向域内单元** 的有向边，出流则相反；墙体边界不加 ghost cell。

流量过程线 $Q(t)$ [m³ s⁻¹] 先除以它穿过的那条边的长度，变成单宽流量 [m² s⁻¹]（与数值方法一致），再作为 ghost cell 的动态特征逐步喂入；水位边界则直接在 ghost cell 上施加已知值。这样 **时变边界条件** 通过图结构自然进入消息传递，而 SWE-GNN 做不到这一点。

{ATTR}
""", ["f03"]))
new.append(rehead(E[9], "#### 在代码/数据中：DEM、ghost cell 对应的入流位置与 20 组测试 hydrograph"))
new.append(E[10])

new.append(md(r"""### 2.4 旋转不变输入（§2.4）

模型的输出（$h$、$|q|$）都是标量，所以作者刻意 **不使用任何依赖坐标方向的特征**：不用坡度的 $x/y$ 分量、不用边的方向、动态输入用单宽流量的 **模** 而不是矢量分量，边特征只有对偶边长。于是输入整体旋转不改变输出（rotation invariance），模型不必从数据里"学会"旋转对称性，样本效率更高。消融实验（Table 3 "rotation-dependent inputs"）表明改用方向相关的输入会使 test RMSE 从 0.052 升到 0.061 m。
"""))

new.append(fig_md(rf"""### 2.5 损失函数与课程学习（§2.5、Appendix B，论文 Fig. 13）

**多步预报损失（Eq. 9）**

$$\mathcal{{L}}_f = \frac{{1}}{{HO}}\sum_{{\tau=1}}^{{H}}\sum_{{o=1}}^{{O}}\gamma_o\,\big\|\hat{{u}}^{{t+\tau}}_o - u^{{t+\tau}}_o\big\|^2,\qquad \gamma_1 = 1\ (h),\ \gamma_2 = 7\ (|q|)$$

- $H$ 是训练时的 rollout 步数，采用 **课程学习**：从 1 步逐渐增加到 $H=6$ 步（对应 12 h），让模型学会修正自己的误差累积；
- 代码中 `only_where_water` 只在有水的单元上计算损失，避免大片干区主导梯度。

**质量守恒项（Appendix B, Eq. B1）**：$\mathcal{{L}}_c = \sum_i a_i\,\Delta\hat h_i - Q\,\Delta t$，即域内水量变化应等于入流体积；总损失 $\mathcal{{L}} = \mathcal{{L}}_f + \alpha\,\mathcal{{L}}_c$。

![Fig 13]({att('f13')})

*Fig. 13 —* 在 $[10^{{-8}}, 5\times10^{{-5}}]$ 内对数均匀采样 $\alpha_m$，验证损失与 CSI$_{{0.05}}$ 没有统计显著的改善（$p$ = 0.42 / 0.48），所以最终模型 $\alpha = 0$（Table D1 加粗）。第 5 节的复现代码会从 `results/mass_conservation.csv` 重画这张图。

{ATTR}
""", ["f13"]))

# =============================================================================== 3 Experimental setup
new.append(fig_md(rf"""## 3. Experimental setup

### 3.1 合成数据集（§3.1，论文 Fig. 4、Fig. 6、Table 1）

- **数值模型**：Delft3D FM（D-HYDRO Suite 1D2D），**100 场** 溃堤洪水模拟：60 训练 / 20 验证 / 20 测试。
- **计算域**：随机生成的椭圆状多边形；**DEM** 由 Perlin 噪声叠加一个随机方向的小坡度生成；4 个网格尺度；Manning 糙率空间均匀 $0.023\ \mathrm{{m^{{-1/3}}\,s}}$。
- **边界条件**：在随机一条边界边上施加入流 hydrograph，形状为 Weibull 型（右偏，符合溃口流量特征），峰值 150–300 m³ s⁻¹。
- **时间**：分辨率 2 h，总时长 96 h，即 48 步。

![Fig 4]({att('f04')})

*Fig. 4 —* 合成数据集中一场模拟的最细网格与 DEM，左下角红圈是 ghost cell 的位置（入流边界）。高程范围只有约 −3 到 +1 m，即近乎平坦的圩田地形。

**Table 1 — 训练 / 验证 / 测试集统计（最细网格，mean ± SD）**

| | Training | Validation | Testing |
|---|---|---|---|
| 高程 [m] | 0.07 ± 0.11 | 0.06 ± 0.10 | 0.07 ± 0.11 |
| 单元数 | 1621 ± 310 | 1608 ± 287 | 1650 ± 320 |
| 单元面积 [m²] | 1922 ± 1161 | 1957 ± 1210 | 1875 ± 1137 |
| 边长 [m] | 48.6 ± 16.2 | 49.2 ± 16.9 | 48.1 ± 15.8 |
| 总洪水体积 [10⁶ m³] | 0.71 ± 0.34 | 0.70 ± 0.31 | 0.72 ± 0.36 |

![Fig 6]({att('f06')})

*Fig. 6 —* 训练（蓝）、合成测试（橙）与 dike ring 15 测试（绿）hydrograph 的均值 ± 1 SD（虚线为极值）。训练与合成测试分布一致；dike ring 15 的流量高 3–5 倍、且尾部不归零（总体积约为合成集的 9 倍），是典型的 **分布外** 迁移场景。

{ATTR}
""", ["f04", "f06"]))
new.append(rehead(E[11], "#### 在数据中：test sim 0 的真值水深随时间演化"))
new.append(E[12])

new.append(fig_md(rf"""### 3.2 真实案例：dike ring 15（§3.2，论文 Fig. 5）

![Fig 5]({att('f05')})

*Fig. 5 —* 荷兰 dike ring 15（Krimpenerwaard–Lopikerwaard，Rotterdam 与 Utrecht 之间，EPSG:28992）。面积 31 400 ha，人口 201 500，单次洪水期望损失 51 亿欧元。沿堤圈大致等距选了 **11 个溃口位置**：红叉 1 处用于微调（训练 + 验证），蓝叉 10 处用于测试。

- 简化：删除水体和 DEM 之外的所有基础设施，糙率均匀。
- hydrograph：先上升（溃口扩展）后缓慢下降、末端不归零，峰值 700–1000 m³ s⁻¹。
- 网格：最细尺度 22 880 个单元（代码中每个样本约 30 000 节点，见 4.2 节）；根据溃口位置不同，盆地呈"浴缸式"或"坡面式"响应。

{ATTR}
""", ["f05"]))

new.append(md(r"""### 3.3 归一化（§3.3）

只对 **面积** 和 **边长** 做 z-score 标准化，且 **每个尺度分别** 统计训练集的均值/方差（粗尺度单元面积大几个数量级，混在一起标准化没有意义）。其余变量（高程、水深、流量）不做归一化。代码里对应 `config.scalers`，由 `create_model_dataset` 在训练集上拟合（本 notebook 只用它变换测试数据）。
"""))

new.append(md(r"""### 3.4 训练设置（§3.4，论文 Table D1）

| 项目 | 取值 |
|---|---|
| 框架 | PyTorch 2.0.1 + PyTorch Geometric 2.4；NVIDIA A100 80 GB |
| 优化器 / 学习率 | Adam，初始 0.003，每 20 epochs ×0.7 |
| epochs | 200，early stopping；16-bit 混合精度；梯度裁剪阈值 1 |
| 课程学习 | 训练 rollout 步数逐渐增加到 $H=6$ |
| 训练时长 | mSWE-GNN 2–15 h，SWE-GNN 5–30 h；dike ring 15 微调约 20 min（减少 epochs 可到 5 min） |

**Table D1 — 超参数搜索范围（加粗 = 验证损失最优）**

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

对照 repo：`results/Pareto_front/models/K{L}_F{G}.h5` 正是 mSWE-GNN 这 4 × 4 = 16 个组合；`config.yaml` 里 `models.K` = $L$，`models.hid_features` = $G$。
"""))
new.append(rehead(E[13], "#### 在代码中：加载预训练 `K4_F64` 并做自回归 rollout（本 notebook 不训练）"))
new.append(E[14])

new.append(md(r"""### 3.5 评价指标（§3.5）

- **MAE / RMSE**（Eq. 10）：整个 rollout 上逐单元、逐时刻的误差，对水深 $h$ 与单宽流量 $|q|$ 分别计算。
- **CSI**（critical success index，Eq. 11）：$\mathrm{CSI}=\dfrac{TP}{TP+FP+FN}$，以水深阈值 0.05 m（"是否被淹"）和 0.3 m（"是否危险水深"）把预测与真值二值化后计算，1 = 淹没范围完全吻合。
- **speed-up**：数值模型耗时 / 深度学习模型推理耗时；深度学习模型对所有测试模拟 **并行** 推理（Appendix C 讨论 batch size 的影响）。

**本 notebook 的口径**：
""" + E[15].source.split("\n", 1)[1]))

# =============================================================================== 4 Results
new.append(md(r"""## 4. Results

### 4.1 合成测试集：与 SWE-GNN 的比较（§4.1）

#### 4.1.1 在代码中：`K4_F64` 的全局指标，与 repo 记录的 `overview_MSGNN.csv` 对比
"""))
new.append(E[16])

new.append(fig_md(rf"""#### 4.1.2 指标随 rollout 时间的演化（论文 Fig. 8）

![Fig 8]({att('f08')})

*Fig. 8 —* 合成测试集上 (a) CSI$_{{0.05}}$ / CSI$_{{0.3}}$ 与 (b) $h$、$|q|$ 的 MAE 随预报时长的变化（阴影 = ±1 SD）。CSI$_{{0.05}}$ 在 48 步内始终高于 0.78；CSI$_{{0.3}}$ 前 20 h 偏低是因为初期深水区很小、几个单元的误差就占很大比例。水深 MAE 随时间单调累积（自回归误差传播），流量 MAE 在洪峰附近最大、随流量减小而回落。下面用 `SpatialAnalysis` 复现这两张图。

{ATTR}
""", ["f08"]))
new.append(rehead(E[17], "##### 在代码中：CSI 与 MAE 随 rollout 时间的变化"))
new.append(E[18])
new.append(rehead(E[19], "#### 4.1.3 在代码中：逐场模拟的误差排名"))
new.append(E[20])
new.append(rehead(E[21], "#### 4.1.4 在代码中：单场模拟可视化"))
new.append(E[22]); new.append(E[23])
new.append(rehead(E[24], "##### 水深与流量在若干时刻的对比（best 模拟）"))
new.append(E[25])
new.append(fig_md(rf"""##### 论文中的对应图：合成测试集上的单宽流量 rollout（Appendix A，论文 Fig. 12）

![Fig 12]({att('f12')})

*Fig. 12 —* 一场合成测试模拟的 $|q|$（对数色标）：真值（上）、预测（中）、差值（下）。模型正确复现了洪水前锋随时间向域内推进又随入流减弱而消退的过程；最大误差出现在溃口附近和 48 h 前后洪峰经过的位置，量级 0.02 m² s⁻¹。上面 `compare_v_rollout` 画的就是同类图。

{ATTR}
""", ["f12"]))
new.append(rehead(E[26], "##### 淹没到达时间 (Flood Arrival Time) 与质量守恒"))
new.append(E[27]); new.append(E[28])
new.append(rehead(E[29], "##### 多尺度视角"))
new.append(E[30])

new.append(fig_md(rf"""#### 4.1.5 Pareto front：速度 vs. 精度（论文 Fig. 7、Table A1）

![Fig 7]({att('f07')})

*Fig. 7 —* mSWE-GNN（圆点）与 SWE-GNN（叉）在 speed-up 对 (a) 验证 RMSE、(b) 验证 CSI$_{{0.05}}$ 平面上的分布，颜色 = 参数量，红色虚线 = 各自的 Pareto front。要点：
- mSWE-GNN 的 Pareto front **整体优于** SWE-GNN——同样精度下快 2–4 倍，同样速度下 RMSE 低约 0.04 m；
- 最快的 mSWE-GNN 变体加速超过 **1200 倍**；虽然它的参数量不比 SWE-GNN 少，但 **最细尺度（节点/边最多）上的层数只有 2–5 层**，而 SWE-GNN 需要 10–18 层，计算量差在这里；
- 层数少也让训练更稳定（gap 2）。论文选定的参考模型是 $L=4, G=64$（速度与精度的折中）。

**Table A1 — 运行时间与加速比（论文选定模型）**

| 数据集 | 数值模型 | mSWE-GNN | speed-up |
|---|---|---|---|
| 合成测试集（20 场） | 12 h 20 min | 0.61 ± 0.02 s | 728 ± 32 |
| dike ring 15（10 场） | 47 h 15 min | 0.24 ± 0.01 s | 708 ± 24 |

下面对 repo 自带的 16 个 checkpoint 逐个 rollout，复现这张 Pareto 图（论文用验证集，这里用测试集）。

{ATTR}
""", ["f07"]))
new.append(rehead(E[31], "##### 在代码中：16 个 checkpoint（K ∈ {2,3,4,5} × F ∈ {16,32,50,64}）的 Pareto front 复现"))
new.append(E[32]); new.append(E[33]); new.append(E[34])

new.append(fig_md(rf"""### 4.2 迁移到真实案例：dike ring 15（§4.2，论文 Fig. 9–11、Table 2）

只在合成数据上训练的模型直接用于 dike ring 15 时，CSI$_{{0.05}}$ 只有 63 %（hydrograph 分布外，见 Fig. 6）；用 **1 场** 模拟微调约 20 min 后，10 个测试溃口的 CSI$_{{0.05}}$ 升到 **87.68 %**，水深 MAE 从 0.31 m 降到 0.12 m。

**Table 2 — 微调的效果（10 场测试，mean ± SD，最细网格）**

| Fine-tuning | MAE $h$ [10⁻² m] ↓ | MAE $\lvert q\rvert$ [10⁻² m² s⁻¹] ↓ | CSI$_{{\tau=0.05\,\mathrm{{m}}}}$ [%] ↑ | CSI$_{{\tau=0.3\,\mathrm{{m}}}}$ [%] ↑ |
|---|---|---|---|---|
| No | 31.09 ± 5.42 | 3.37 ± 1.24 | 63.36 ± 19.54 | 46.06 ± 18.62 |
| Yes | **12.07 ± 4.19** | **2.08 ± 0.82** | **87.68 ± 10.3** | **81.82 ± 16.07** |

![Fig 9]({att('f09')})

*Fig. 9 —* dike ring 15 一个测试溃口（左上红叉）的水深 rollout：真值（上）、预测（中）、差值（下）。模型抓住了洪水沿低洼圩田向东北扩展的整体动态；误差主要是 48 h 之后系统性略高估（紫色）以及溃口附近的局部低估。

![Fig 10]({att('f10')})

*Fig. 10 —* 同一场的淹没到达时间（FAT，水深首次超过 0.05 m 的时刻）：预测的到达时间场与真值几乎一致，差异集中在洪水前锋末端（±12–24 h）。FAT 是应急撤离最关心的量。

![Fig 11]({att('f11')})

*Fig. 11 —* 微调后的模型在 10 个测试溃口上的 CSI$_{{0.05}}$：0.82–0.94，说明只用一个溃口微调就能泛化到堤圈上其它位置（不同的入流方向与盆地响应）。

{ATTR}
""", ["f09", "f10", "f11"]))
new.append(rehead(E[37], "#### 在代码中：加载 `finetuned_dk15.h5`，在 dijkring 15 的 10 场测试上 rollout"))
new.append(E[38]); new.append(E[39])

new.append(md(r"""### 4.3 消融实验（§4.3，论文 Table 3）

| 配置 | Val RMSE [m] | Val CSI$_{0.05}$ | Test RMSE [m] | Test CSI$_{0.05}$ |
|---|---|---|---|---|
| **mSWE-GNN（完整）** | **0.044** | **0.956** | **0.052** | **0.943** |
| w/o multi-scale module | 0.051 | 0.948 | 0.061 | 0.929 |
| learnable downsampling（代替 mean pooling） | 0.048 | 0.950 | 0.056 | 0.936 |
| w/o skip connections（Eq. 7） | 0.045 | 0.955 | 0.053 | 0.941 |
| residual decoder（代替 Eq. 8 的 1D CNN） | 0.046 | 0.954 | 0.054 | 0.940 |
| rotation-dependent inputs | 0.051 | 0.947 | 0.061 | 0.928 |

*数值转录自文章正文，使用前请对照原表。* 结论：**多尺度模块** 和 **旋转不变输入** 贡献最大（去掉后 test RMSE 各上升约 17 %）；skip connection 与 decoder 形式的影响较小但方向一致。这与第 1 节的 gap 对应：多尺度解决传播速度差异与深层不稳定，旋转不变性提高样本效率。
"""))

# =============================================================================== 5 Discussion / Conclusion / Appendix C
new.append(fig_md(rf"""## 5. Discussion 与 Conclusion（§5–6）+ Appendix C

**优势（对应第 1 节的四个 gap）**
1. 多尺度 U 形结构用每尺度 2–5 层就覆盖整个域 → 能表达快慢不同的传播，训练稳定，最细尺度计算量小 → 加速 700–1200 倍，Pareto front 全面优于 SWE-GNN；
2. ghost cell + 有向边 → 时变的溃口 hydrograph 可作为输入；
3. 从干河床 $t=0$ 开始 rollout → 不需要数值求解器给初始条件；
4. 旋转不变输入 + 图结构 → 可迁移到未见过的网格、地形和边界条件，真实案例只需 1 场模拟微调。

**局限（作者自述）**：只评估了溃堤洪水（未考虑河流/海岸/暴雨型洪水）；未测试时变水位边界；糙率空间均匀；网格必须自顶向下由边界多边形生成，不能直接用现成的细网格；未考虑多个同时的边界条件；各尺度层数取相同值；未与 FNO、neural fields 等新方法比较。

**未来工作**：时变溃口扩展模型；加入水体与线状要素（道路、次级堤防）；降雨输入与 1D 排水耦合（城市内涝）；概率式多情景框架与不确定性量化；PINN 式自动微分损失；更大训练集以省去微调；JIT/IPU 加速。

### Appendix C：并行推理的加速（论文 Fig. 14）

![Fig 14]({att('f14')})

*Fig. 14 —* 16 个 Pareto 模型在合成测试集上的 speed-up 随 batch size（同时推理的模拟数）的变化（双对数）。并行 20 场模拟可再获得约 4.5 倍加速；层数多（大圆点）、参数多（深色）的模型受益更小。若把微调所需的那一场数值模拟的时间也算进去，真实案例的总体加速约为 4–8 倍。

{ATTR}
""", ["f14"]))
new.append(rehead(E[35], "#### 在代码中：repo 记录的并行推理耗时与质量守恒实验（不重跑，直接读 csv）"))
new.append(E[36])

# =============================================================================== 6 reproduction notes
new.append(rehead(E[40], "## 6. 复现结论与集群上的注意事项"))

nb.cells = new
nbf.validate(nb)
nbf.write(nb, DST)

n_code = sum(c.cell_type == "code" for c in new)
n_code_src = sum(c.cell_type == "code" for c in E)
n_att = sum(len(getattr(c, "attachments", {}) or {}) for c in new)
print(f"wrote {DST}: {len(new)} cells ({n_code} code, was {n_code_src}), {n_att} embedded figures")
assert n_code == n_code_src
