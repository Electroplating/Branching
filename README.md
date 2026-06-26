# OMT 分支选择 GNN 策略 (omt_branching)

基于 [`plan.md`](plan.md) 实现的图神经网络 OMT 分支选择策略框架。代码按职责分为
**输入 / 模型 / 输出** 三个相互解耦的部分，每一部分都有明确的接口契约，方便与
Z3/νZ、OptiMathSAT 等求解器集成。

## 总体数据流

```
求解器 ──SolverSnapshot──▶ 输入(graph_builder) ──HeteroGraph──▶ 模型(policy/inference)
                                                                       │
求解器 ◀──BranchingAdvice──── 输出(decoder) ◀──PolicyOutput───────────┘
```

- 求解器只需按 `omt_branching.input.solver_state` 中的 dataclass 填充一次快照。
- 输出 `BranchingAdvice` 定义在 `omt_branching.output.advice`，规定了返回给求解器
  的字段（activity 先验、候选排序、phase 建议、整数 B&B split、置信度与回退标记）。
- 模型部分不感知求解器内部结构，只在 `HeteroGraph` 与 `PolicyOutput` 上工作。

## 目录结构

```
omt_branching/
├── interfaces.py            # 通用枚举与常量（节点/边/原子类型）
├── graph/hetero_graph.py    # 轻量异构图容器（不依赖 PyG）
├── input/
│   ├── solver_state.py      # 【输入接口】求解器需提供的信息类型 (dataclass schema)
│   └── graph_builder.py     # SolverSnapshot -> HeteroGraph
├── model/
│   ├── gnn.py               # 异构关系消息传递 GNN (R-GCN 风格)
│   ├── heads.py             # branching / phase / B&B / 辅助任务 ranking head
│   ├── policy.py            # 编码器 + 多头组合的策略网络
│   ├── trainer.py           # 阶段一：离线 imitation learning
│   ├── finetune.py          # 阶段二：solver-in-the-loop 微调 (DAgger/REINFORCE)
│   └── inference.py         # 部署期推理（含预算控制与 OOD 回退）
└── output/
    ├── advice.py            # 【输出接口】返回给求解器的信息格式 (dataclass schema)
    └── decoder.py           # PolicyOutput -> BranchingAdvice
```

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
python -m examples.demo
```

`examples/demo.py` 构造一个合成的 OMT 求解器快照，依次跑通建图、推理、解码，
并演示一步 imitation 训练，可作为集成模板。

## 与求解器集成（最小可行路径）

参考 `plan.md` 第 7、12 节，推荐 **VSIDS refocus** 模式：

1. 在初始化 / restart 后 / 每 N 次 conflict，按 `solver_state` 填一个
   `SolverSnapshot` 并调用 `BranchingPolicyService.advise(snapshot)`。
2. 求解器读取返回的 `BranchingAdvice.activity_priors`，把先验混合进 SAT activity；
   普通 decision 仍由原生 priority queue 决定。
3. 若 `advice.fallback is True`（超时 / 低置信 / OOD），求解器使用原生 heuristic。

## 实验与评测

项目已包含完整的合成数据实验流程，详见 [`experiments/EXPERIMENTS.md`](experiments/EXPERIMENTS.md)。

快速复现：

```bash
conda activate omt
python -m examples.demo
python -m experiments.run_experiments
python -m experiments.analyze_results
python -m experiments.plot_results
```

主要结论：

- GNN 在合成 OMT 快照上的 top-1 准确率（0.43–0.73）显著高于 VSIDS baseline（0.067）。
- 增加 GNN 深度（4 层）比增加宽度更有效，最佳 top-1 达到 0.733。
- 跨分布泛化（训练小图、测试大图）仍有挑战，需要 size-agnostic 设计或真实 solver 数据。
- 单次推理开销约 12 ms（500 变量规模），部署时可接受，但真实 solver 需要子图采样与 refocus 策略。

## 后续优化方向

- 模型：引入 attention / Graph Transformer、edge gating、global readout。
- 训练：hard negative mining、损失权重搜索、DAgger/REINFORCE 在线微调。
- 部署：周期性 refocus、root embedding 缓存 + 轻量 MLP、k-hop 子图采样、OOD 回退。
- 集成：在 Z3/νZ/OptiMathSAT 中插桩采集真实轨迹，验证 wall-clock time 与 PAR-2 改进。
