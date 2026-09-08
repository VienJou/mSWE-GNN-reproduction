"""Insert the receptive-field / compute-cost analysis into test_pretrained.ipynb.

Adds two markdown cells (no code, no re-execution needed):
  - §2.2.1 after the network-architecture cell: per-scale hop distances measured from
    the raw pickles, U-path receptive field, cost in finest-scale-layer equivalents,
    and the correction that one forward pass does NOT span the whole domain.
  - §5.1 at the end of the discussion: what that split (multi-scale vs autoregression)
    means for a static single-pass raster surrogate such as our Dataset_v4 model.

    python plan/add_receptive_field_section.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "test_pretrained.ipynb"

nb = nbf.read(NB, as_version=4)
cells = nb.cells

SEC_221 = r"""### 2.2.1 感受野与计算量：为什么"每尺度 4 层"就够

上面说"粗尺度让信息在少量 GNN 层内跨越整个计算域"，这句话需要算清楚——它对了一半。

**前提：细化规则决定了尺度间的固定比例。** 网格细化是"每条边一分为二"，所以相邻尺度的 **特征长度差 2 倍、单元数差 4 倍**。用上一格 cell 打印的 test sim 0 和 §4.2 的 dijkring 15 实测（跳距 = 相邻单元重心距 `face_distance` 的中位数，取自未经归一化的 pickle）：

| 尺度 | 合成 test sim 0 单元数 | 跳距 | dijkring 15 单元数 | 跳距 |
|---|---|---|---|---|
| $\mathcal{M}_1$ 最细 | 11 837 | 91 m | 22 881 | 98 m |
| $\mathcal{M}_2$ | 2 961 (÷4.00) | 185 m (×2.02) | 5 724 (÷4.00) | 198 m (×2.02) |
| $\mathcal{M}_3$ | 741 (÷4.00) | 371 m (×2.00) | 1 433 (÷3.99) | 396 m (×2.00) |
| $\mathcal{M}_4$ 瓶颈 | 186 (÷3.98) | 739 m (×1.99) | 359 (÷3.99) | 785 m (×1.98) |

比例是严格的 4 倍与 2 倍。§4.2 打印的 `node_ptr [0, 22881, 28605, 30038, 30397]` 也正是这个关系：$22\,881\times(1+\tfrac14+\tfrac1{16}+\tfrac1{64}) \approx 30\,400$，即"30 397 个节点"是四个尺度节点数之和，最细尺度本身只有 22 881 个。

**感受野沿 U 形路径累加。** 每个尺度的 GNN 模块做 $K=4$ 跳（`config.yaml` 的 `models.K`；代码里 `self.K = [K]*4 + [K]*3`，共 7 个模块），跳距按尺度翻倍：

$$\text{RF} = K\,(h_1 + 2h_1 + 4h_1 + 8h_1 + 4h_1 + 2h_1 + h_1) = 4\times 22\,h_1 = 88\,h_1$$

代入实测跳距：合成数据集 **8.1 km**，dijkring 15 **8.7 km**。

**对比单尺度。** 要在最细网格上走满同样的 88 跳，需要 **88 层** GNN。SWE-GNN 实际用 10–18 层（Table D1），感受野只有 0.9–1.6 km——这正是论文 gap #1 和 gap #2 的量化形式。

**代价省在哪里。** 7 个模块 × 4 跳 = 28 层，但其中只有 8 层（下行 4 + 上行 4）跑在最细网格上，其余跑在节点数为 1/4、1/16、1/64 的粗网格上。按节点数加权：

$$4\times\Big(1+\tfrac14+\tfrac1{16}+\tfrac1{64}+\tfrac1{16}+\tfrac14+1\Big) = 10.6\ \text{个"最细尺度层"当量}$$

**用约 10.6 层的计算成本，买到 88 层的感受野**（约 8 倍）。这解释了 Fig. 7 里那个看似矛盾的现象：mSWE-GNN 参数量更大（81 万）却更快（加速比至 1200 倍）。反向传播路径也从 88 层缩短到 28 层，这是论文所说"训练更稳定"的机制。

**需要澄清的一点：一次前向并不覆盖全域。** 瓶颈层单跳只有 0.74 km，整条 U 路径的感受野约 8 km，而合成域直径约 13 km、dijkring 15 长边约 36 km。也就是说单次前向分别只覆盖约 63 % 和 24 % 的域。

域尺度的耦合来自 **自回归**：48 个时间步滚下来，累积感受野是 $48\times 8 \approx 380$ km，远超任何实际域。信息像波一样逐步传播，而不是一次前向就全域可见。所以 mSWE-GNN 是用 **两个机制** 共同承载长程依赖：
- **多尺度** 让单步传播距离与真实洪水波速匹配（2 h 步长内洪水推进 km 量级，正好对应 8 km 的单步感受野），避免用几十层去追一个时间步；
- **自回归** 把域尺度的耦合摊到 48 步里。

这两半的分工在 §5.1 还会用到——它决定了这套方法里哪一部分可以迁移到静态、单次推理的模型上。
"""

SEC_51 = r"""### 5.1 迁移到静态单次推理的栅格模型（我们 Dataset_v4 的情形）

把 §2.2.1 的结论接到我们自己的溃坝淹没模型（`Dataset_v4/runs/hurdle_v4nat_seed42`，HurdleUNet，19 通道栅格 U-Net）上，需要先分清 mSWE-GNN 的两个机制哪个可迁移。

**两者的长程依赖结构不同**

| | mSWE-GNN | 我们的 HurdleUNet |
|---|---|---|
| 任务 | 时序演进 $h(t), q(t)$，48 步 | 静态最大淹没包络，单次推理 |
| 单次前向感受野 | 8.1 km（域的 63 %） | 512 px crop 封顶，60 m/px 下 30.7 km |
| 域尺度 | 合成 13 km / dijkring 36 km | metric canvas 长边中位数 159 km，432/467 超过 50 km |
| 跨局部耦合 | 多尺度（每步）+ 自回归（跨步） | 无：tile 独立推理，重叠区只做概率平均 |
| 长程信息来源 | 架构 | 手工通道 `flow_dist`、`q_scn`、`stage_scn`、`dist_dam`、`dam_heat` |

**可迁移的是多尺度那一半。** 给 U-Net 加一个粗分辨率的全局分支（把整张 canvas 降采样后编码，与细尺度 crop 的 bottleneck 特征拼接），对应的正是 mSWE-GNN 的多尺度机制，原则上可以替代甚至取代那几个手工沿程特征。

**不可迁移的是自回归那一半。** 我们预测的是最大淹没包络，没有时间维，因此没有任何机制能把耦合摊到多步里；全局上下文只能由架构一次性提供。对静态任务而言这可能恰好是对的取舍，但必须写清楚，否则"加个粗分支就等价于 mSWE-GNN"是错的。

**顺带一个引用上的提醒：不要直接对标 700 倍加速。** 那个加速比的分母是一次 48 步的 Delft3D 时序模拟；我们的标签是 USACE FIM 已发布的静态成果图，替代的是一整个建模研究而不是一次求解器运行，语境不同。同理 CSI 数值也不可直接比（CSI 与二值 IoU 是同一个量，但 mSWE-GNN 的 0.88 是单一堤圈、经过微调、且真值来自同一求解器；我们的 0.45 是全国范围、未见过的大坝、真值来自第三方发布图）。
"""


def find(prefix):
    hits = [i for i, c in enumerate(cells)
            if c.cell_type == "markdown" and c.source.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


# guard against a double run
assert not any(c.source.startswith("### 2.2.1") for c in cells), "already inserted"

i_arch = find("### 2.2 网络结构")
i_sec6 = find("## 6. 复现结论")

cells.insert(i_sec6, nbf.v4.new_markdown_cell(SEC_51))
cells.insert(i_arch + 1, nbf.v4.new_markdown_cell(SEC_221))

nb.cells = cells
nbf.validate(nb)
nbf.write(nb, NB)
print(f"wrote {NB}: {len(cells)} cells "
      f"(+2 markdown; §2.2.1 at {i_arch + 1}, §5.1 at {i_sec6 + 1})")
