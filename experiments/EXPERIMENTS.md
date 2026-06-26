# OMT 分支选择 GNN 策略实验报告

## 1. 环境设置

使用 conda 从已有 `llm` 环境克隆出专用环境 `omt`，避免重复下载 PyTorch/CUDA 包：

```bash
conda create --name omt --clone llm -y
conda activate omt
cd /mnt/d/D_Work/ISCAS/paper/AAAI2027/NeuralOMT/Branching
python -m examples.demo          # 验证基础流程
python -m experiments.run_experiments  # 完整实验
```

- Python: 3.10（omt 环境）
- PyTorch: 2.7.1+cu126
- GPU: NVIDIA GeForce RTX 4060 Laptop
- CUDA: 12.6

## 2. 数据集

由于尚未接入真实 Z3/νZ/OptiMathSAT 求解器，使用 `experiments/synthetic_omt.py` 生成合成 OMT(LIA) 快照：

| 参数 | 默认值 |
|------|--------|
| 布尔变量数 `n_bool` | 30 |
| 整数变量数 `n_numeric` | 12 |
| CNF 子句数 `n_clauses` | 60 |
| 理论原子数 `n_atoms` | 20 |
| soft clause 比例 `soft_ratio` | 0.15 |

每个快照包含：布尔变量（赋值/候选/VSIDS/LRB/CHB/phase）、CNF 子句、理论原子（线性不等式）、整数变量（LP 松弛/界/pseudo-cost）、目标函数与全局搜索状态。

## 3. 专家标签与 Baseline

专家标签由 `experiments/oracle.py` 生成，综合以下 OMT-aware 信号：

- VSIDS / LRB activity
- 理论原子的 tightness / violation
- 与 objective 系数大的数值变量的关联
- soft clause 权重
- 出现次数与 recent learned clause 参与

分数经过 softmax 归一化，作为 imitation ranking 的软标签。

对比 Baseline：

- **VSIDS**：仅使用 VSIDS activity，与 Oracle 几乎无关（top-1 ≈ 0.067）。
- **Oracle heuristic**：直接使用专家规则选择，top-1 = 1.0（作为理论上限）。

## 4. 实验设计

使用 `experiments/run_experiments.py` 运行以下实验：

| 实验 | 训练规模 | 模型配置 | 备注 |
|------|----------|----------|------|
| baseline | 120 train / 30 val | hidden=64, layers=3 | 默认 R-GCN 风格 GNN |
| deep_gnn | 120 train / 30 val | hidden=64, layers=4 | 测试深度影响 |
| wide_gnn | 120 train / 30 val | hidden=128, layers=3 | 测试宽度影响 |
| generalization_large | 120 train（小图） / 30 val（大图） | hidden=64, layers=3 | 跨分布泛化 |

训练配置：

- Epochs: 20
- LR: 1e-3
- Gradient accumulation: 4 steps
- Loss: `L_branch + 0.5*L_phase + 1.0*L_int + 0.3*L_aux`

评估指标：

- `gnn_top1` / `gnn_top3`：候选布尔变量 ranking 的 top-1/top-3 准确率
- `gnn_phase`：phase 预测准确率
- `gnn_int_top1`：整数 B&B split 候选 top-1 准确率

## 5. 实验结果

| 实验 | top1 | top3 | phase | int1 | vsids | oracle |
|------|------|------|-------|------|-------|--------|
| baseline | 0.567 | 0.833 | 1.000 | 0.867 | 0.067 | 1.000 |
| deep_gnn | **0.733** | 0.867 | 1.000 | 0.800 | 0.067 | 1.000 |
| wide_gnn | 0.600 | 0.800 | 1.000 | 0.833 | 0.067 | 1.000 |
| generalization_large | 0.433 | 0.767 | 1.000 | 0.800 | 0.067 | 1.000 |

运行时间：

- baseline: ~149 s
- deep_gnn: ~232 s
- wide_gnn: ~148 s
- generalization_large: ~225 s

完整结果保存在 `experiments/results/*.json`，可视化图表在 `experiments/results/plots/`。

## 6. 分析与讨论

### 6.1 GNN 显著优于 VSIDS

所有 GNN 变体的 top-1 准确率（0.43–0.73）都远高于 VSIDS（0.067），说明在合成 OMT 快照上，异构 GNN 能够有效整合 theory/objective/soft 等 OMT-aware 特征，而纯 VSIDS activity 无法捕捉这些信息。

### 6.2 模型深度比宽度更有效

`deep_gnn`（4 层）top-1 达到 0.733，优于 `wide_gnn`（128 hidden，0.600）和 baseline（0.567）。说明在这个任务上，增加消息传递深度比单纯增加 hidden size 更能提升表示能力。

### 6.3 跨分布泛化仍有挑战

`generalization_large`（训练用小图、测试用 2 倍大图）top-1 降至 0.433。这符合预期：合成数据分布变化后，模型需要更好的 size-agnostic 设计（如 positional encoding、子图采样、或按 instance family 分桶训练）。

### 6.4 与 Oracle 仍有差距

即使最好的 `deep_gnn`（0.733）也低于 Oracle（1.0）。原因包括：

1. 标签是软概率分布，模型学习的是 ranking 而非精确 argmax。
2. 损失函数中多任务权重可能不是最优。
3. 训练数据规模小（120 个 snapshots），且 synthetic 数据带有随机噪声。
4. 模型未使用 attention / global readout 等更强的结构。

### 6.5 推理开销

`experiments/profile_inference.py` 测量显示，即使 scale=500（约 2000 节点 / 5000 边），单次 forward 仅需约 12 ms。说明当前 GNN 在中小规模实例上的推理开销可控，但真实 solver 的图可能大 1–2 个数量级，仍需规模门限与 k-hop 子图采样。

## 7. 后续优化方向

### 7.1 模型结构

- **Attention / Transformer**：在关系消息传递中加入 cross-type attention 或 Graph Transformer，增强全局状态到候选节点的注入。
- **Edge feature gating**：用边特征对消息进行门控，更好地区分不同系数/符号的 variable-atom 关系。
- **Global readout + candidate scoring**：先做一次全局 readout，再与候选节点表示拼接，提升 ranking 质量。

### 7.2 训练策略

- **损失重加权 / 归一化**：当前 aux MSE 目标已归一化，但 branch ranking 与 auxiliary 损失的相对权重可进一步搜索。
- **Hard negative mining**：在 ranking loss 中加大对 top-2/top-3 错误排序的惩罚。
- ** larger & cleaner dataset**：接入真实 Z3/νZ/OptiMathSAT 日志，或生成按 family 控制分布的合成数据。
- **DAgger / REINFORCE**：在 imitation 基础上用 solver-in-the-loop 微调修正分布偏移。

### 7.3 部署优化

- **周期性 refocus**：不要每个 decision 都调用 GNN，只在 restart 后或每 N 次 conflict 运行一次，输出 activity prior。
- **Root embedding 缓存 + 轻量 MLP**：深层节点使用缓存的 root embedding 和当前动态特征，通过 MLP 快速打分。
- **k-hop 子图采样**：图太大时只抽取候选变量 k-hop 子图、recent conflict 子图和 objective-relevant 子图。
- **OOD 检测与回退**：当 confidence 低于阈值或图规模超限时自动回退原生 VSIDS/LRB。

### 7.4 真实求解器集成

- 在 Z3/νZ 中插桩导出 `SolverSnapshot`。
- 实现 VSIDS refocus 模式：把 GNN 输出的 `activity_priors` 按权重混入 solver activity。
- 采集真实决策轨迹，用于在线微调和跨 family 泛化评估。

## 8. 复现命令

```bash
conda activate omt
cd /mnt/d/D_Work/ISCAS/paper/AAAI2027/NeuralOMT/Branching

# 快速验证（约 1 分钟）
python -m experiments.run_quick_experiments

# 完整实验（约 12 分钟）
python -m experiments.run_experiments

# 单独运行某个实验
python -m experiments.run_experiments deep_gnn

# 推理开销分析
python -m experiments.profile_inference

# 求解器模拟对比
python -m experiments.sim_compare

# 结果分析与可视化
python -m experiments.analyze_results
python -m experiments.plot_results
```
