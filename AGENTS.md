# AGENTS.md —— omt_branching

> 本文件面向 AI 编程助手。项目的主要自然语言是**中文**，代码注释、文档字符串（docstring）以及 `README.md`、`plan.md` 均以中文撰写。修改代码时请保持中文文档与中文注释风格。

---

## 1. 项目概述

`omt_branching` 是一个用于 **OMT（Optimization Modulo Theories）分支选择** 的图神经网络（GNN）策略框架原型。它的目标是把 SMT/OMT 求解器在搜索过程中遇到的决策状态表示成一张异构图，然后用 GNN 预测：

- 下一个应该分支的布尔变量（SAT decision）；
- 该变量的极性/phase（取真还是取假）；
- 整数变量的 B&B split 选择及方向；
- 若干辅助信号（冲突概率、unsat core 成员、目标改善、子树规模等）。

框架本身**不绑定任何具体求解器**，而是通过两个稳定的 dataclass 契约与外部求解器交互：

- **输入契约**：求解器构造 `SolverSnapshot`（见 `omt_branching.input.solver_state`）；
- **输出契约**：框架返回 `BranchingAdvice`（见 `omt_branching.output.advice`）。

推荐集成方式是 **VSIDS refocus**（周期性把 GNN 输出的 activity prior 混入求解器原生 VSIDS/LRB activity），而不是每个 decision 都调用 GNN，从而降低推理开销并保留求解器完备性。

当前仓库是一个纯 Python 库 + 一个端到端示例，**尚未包含真实求解器集成代码**。

---

## 2. 仓库结构

```text
Branching/
├── README.md                # 项目简介、安装、快速开始、集成建议
├── plan.md                  # 完整的研究/实现方案（背景、图设计、训练、评测）
├── requirements.txt         # 依赖：torch>=2.0, numpy>=1.21
├── examples/
│   ├── __init__.py
│   └── demo.py              # 端到端示例：构造合成快照 -> 建图 -> 推理 -> 训练
└── omt_branching/           # 主包
    ├── __init__.py          # 顶层导出：NodeType/EdgeType/...、BranchingPolicyService
    ├── interfaces.py        # 跨模块枚举：节点/边类型、原子/子句/搜索模式
    ├── service.py           # 门面：BranchingPolicyService.advise(snapshot)
    ├── graph/
    │   ├── __init__.py
    │   └── hetero_graph.py  # 轻量异构图容器（不依赖 PyG）
    ├── input/
    │   ├── __init__.py
    │   ├── solver_state.py  # 输入接口 dataclass：SolverSnapshot 及其成员
    │   └── graph_builder.py # SolverSnapshot -> HeteroGraph，含 FeatureSpec
    ├── model/
    │   ├── __init__.py
    │   ├── gnn.py           # R-GCN 风格异构消息传递编码器
    │   ├── heads.py         # branching / phase / integer / auxiliary heads
    │   ├── policy.py        # BranchingPolicy：encoder + heads；PolicyOutput
    │   ├── trainer.py       # ImitationTrainer：离线 imitation learning
    │   ├── finetune.py      # SolverInLoopFinetuner：DAgger / REINFORCE 微调
    │   └── inference.py     # InferenceEngine：规模/时间/置信度门控 + fallback
    └── output/
        ├── __init__.py
        ├── advice.py        # 输出接口 dataclass：BranchingAdvice、IntegerSplitAdvice
        └── decoder.py       # PolicyOutput -> BranchingAdvice
```

关键事实：

- **没有 `pyproject.toml`、`setup.py`、`setup.cfg`、`Makefile`、CI/CD 配置或测试目录**。
- 依赖仅 `torch>=2.0` 和 `numpy>=1.21`，未使用 PyTorch Geometric、Lightning 等框架。
- 代码总量约 1900 行（含示例），属于研究原型阶段。
- Git 远程：`git@github.com:Electroplating/Branching.git`，分支 `main`。

---

## 3. 技术栈与运行时架构

### 3.1 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.9+（推荐；仓库内存在 Python 3.10 生成的 `__pycache__`，但当前环境为 3.12） |
| 深度学习 | PyTorch >= 2.0 |
| 数值计算 | NumPy >= 1.21 |
| 图神经网络 | 手写 R-GCN 风格实现，不依赖 PyG/DGL |
| 配置/契约 | 标准库 `dataclasses` + `enum` |

### 3.2 数据流

求解器一次咨询的完整流程：

```text
SolverSnapshot
    │
    ▼
GraphBuilder.build(snapshot) ──▶ HeteroGraph
    │
    ▼
InferenceEngine.run(graph) ────▶ PolicyOutput
    │
    ▼
AdviceDecoder.decode(out) ─────▶ BranchingAdvice
```

- `SolverSnapshot` 由求解器按 `omt_branching.input.solver_state` 中的 dataclass 填充。
- `GraphBuilder` 把快照编码成 `HeteroGraph`（节点/边按类型分桶的 `torch.Tensor`），同时维护 `id_maps` 把求解器原始 id 映射到局部索引。
- `BranchingPolicy`（encoder + heads）前向得到 `PolicyOutput`：各 head 的未归一化分数、候选 mask、辅助预测。
- `InferenceEngine` 在推理前做**规模门控**（`max_total_nodes`）、**时间预算**（`time_budget_ms`）、**置信度门控**（`min_confidence`）；不满足时返回 `None` 并写入 `graph.meta["inference"]` 诊断信息，触发 fallback。
- `AdviceDecoder` 把图内索引还原成求解器 id，生成 `BranchingAdvice`。

### 3.3 关键设计

- **异构图节点类型**（`NodeType`）：`bool_var`、`clause`、`theory_atom`、`numeric_var`、`objective`、`search_state`。
- **边类型**（`EdgeType`）：literal_in_clause、atom_abstracted_by、variable_in_atom、variable_in_objective、soft_weight、bound_relates_variable、state_to_bool、state_to_objective。
- **特征维度硬编码**：`graph_builder.py` 中 `_NODE_DIMS` 和 `_EDGE_DIMS` 与 `_encode_*` 函数严格对应，`_check` 会在运行时断言维度一致。
- **缺失值编码**：可选标量用 `[value, present_flag]`；三态布尔用 one-hot `[none, true, false]`。
- **数值变换**：使用 `log1p`、符号对数、截断等方式降低尺度差异。
- **候选 mask**：`SolverSnapshot` 可显式给出 `candidate_bool_ids` / `candidate_numeric_ids`，否则由 `candidate_bool_set()` / `candidate_numeric_set()` 自动推断。

---

## 4. 构建与运行命令

### 4.1 安装依赖

```bash
cd /mnt/d/D_Work/ISCAS/paper/AAAI2027/NeuralOMT/Branching
pip install -r requirements.txt
```

`requirements.txt` 内容：

```text
torch>=2.0
numpy>=1.21
```

> 注：当前环境未安装 PyTorch，直接运行示例会报 `ModuleNotFoundError: No module named 'torch'`。请在隔离虚拟环境中安装依赖。

### 4.2 运行示例

```bash
python -m examples.demo
```

`examples/demo.py` 会：

1. 构造一个合成 OMT(LIA) 风格的 `SolverSnapshot`；
2. 用 `GraphBuilder` 建图并打印图摘要；
3. 通过 `BranchingPolicyService.advise()` 得到 `BranchingAdvice`；
4. 演示 `BranchingAdvice.mixed_activity()` 与原生活动的融合；
5. 跑 5 步 imitation 训练；
6. 跑 1 步 REINFORCE 微调。

这是验证端到端流程是否正常的主要方式。

### 4.3 作为包导入

顶层入口为 `BranchingPolicyService`：

```python
from omt_branching import BranchingPolicyService
from omt_branching.input import SolverSnapshot

service = BranchingPolicyService()
advice = service.advise(snapshot)
```

---

## 5. 代码组织与主要模块

### 5.1 `omt_branching.interfaces`

所有模块共享的枚举与 schema：

- `NodeType`、`EdgeType`、`EDGE_SCHEMA`
- `AtomKind`、`ClauseKind`、`SearchMode`

### 5.2 `omt_branching.graph`

`HeteroGraph`：

- `node_features: dict[NodeType, Tensor]`
- `edge_index: dict[EdgeType, Tensor]`（shape `[2, num_edges]`）
- `edge_features: dict[EdgeType, Tensor]`
- `id_maps` / `rev_id_maps`：求解器 id 与局部索引互查
- `to(device)`、`finalize()`、`summary()`

### 5.3 `omt_branching.input`

- `solver_state.py`：定义 `BooleanVarInfo`、`ClauseInfo`、`TheoryAtomInfo`、`NumericVarInfo`、`ObjectiveInfo`、`SearchStateInfo`、`SolverSnapshot`。
- `graph_builder.py`：`GraphBuilder` 把这些 dataclass 编码成 `HeteroGraph`；`FeatureSpec` 暴露节点/边维度。

### 5.4 `omt_branching.model`

- `gnn.py`：`HeteroEncoder` = input projection + `RelationalLayer` × L；聚合用 `index_add_` + mean。
- `heads.py`：`BranchingHead`、`PhaseHead`、`IntegerBranchHead`、`AuxiliaryHeads`。
- `policy.py`：`BranchingPolicy` 把 encoder 和 heads 组合；输出 `PolicyOutput`；含 masked softmax 工具。
- `trainer.py`：`ImitationTrainer` + `RankingExample` + `TrainConfig`；使用 ListNet 风格的 ranking loss、phase BCE、整数 ranking + 方向、辅助任务 MSE/BCE。
- `finetune.py`：`SolverInLoopFinetuner` + `Trajectory`；支持 `dagger_update` 和 `reinforce_update`（带移动平均 baseline 与熵正则）。
- `inference.py`：`InferenceEngine` + `InferenceConfig`；负责规模/时间/置信度门控。

### 5.5 `omt_branching.output`

- `advice.py`：`BranchingAdvice`、`IntegerSplitAdvice`；提供 `top_candidate()` 和 `mixed_activity()`。
- `decoder.py`：`AdviceDecoder` 把 `PolicyOutput` 解码回求解器 id，并决定是否 `use_gnn`。

### 5.6 `omt_branching.service`

`BranchingPolicyService` 是求解器侧应持有的唯一门面；内部串联 `GraphBuilder`、`InferenceEngine`、`AdviceDecoder`。

---

## 6. 开发约定与代码风格

- **文档语言**：模块级、类级、函数级 docstring 使用中文；`__all__` 与类型注解使用英文标识符。
- **导入习惯**：每个文件开头写 `from __future__ import annotations`，广泛使用 `dict`、`list` 等内置泛型。
- **配置即 dataclass**：`PolicyConfig`、`TrainConfig`、`FinetuneConfig`、`InferenceConfig`、`DecoderConfig`、`ServiceConfig`、`FeatureSpec` 均为 frozen 或普通 dataclass。
- **枚举优先**：节点/边/原子/子句/搜索模式均用 `Enum`，并通过 `all()` 类方法保证 one-hot 索引稳定。
- **特征维度校验**：`GraphBuilder._check` 在编码每个节点后断言维度与 `FeatureSpec` 一致；修改特征时必须同步更新 `_NODE_DIMS` / `_EDGE_DIMS`。
- **设备管理**：`HeteroGraph.to(device)`、`BranchingPolicy.to(device)`、`InferenceEngine` 均在配置中接受 `device` 字符串。
- **候选局部索引**：训练标签（`RankingExample`）和模型输出（`PolicyOutput`）均使用**图内局部索引**而非求解器原始 id；解码器负责还原。
- **无 PyG 依赖**：所有图操作手写，保持最小依赖；如需迁移到 PyG/DGL，需要重写 `HeteroEncoder` 与 `HeteroGraph` 的交互层。

---

## 7. 测试说明

- **当前没有单元测试、集成测试或测试目录**。
- 验证方式主要是运行 `python -m examples.demo`，检查：
  - `HeteroGraph.summary()` 输出节点/边数量是否符合预期；
  - `BranchingAdvice` 字段是否被正确填充；
  - imitation 与 REINFORCE 训练步骤是否能正常 backward/更新。
- 新增代码时建议补充的测试方向：
  - `GraphBuilder` 对各种缺失字段（`None`）的健壮性；
  - `HeteroEncoder` 对空图/单边类型的处理；
  - `AdviceDecoder` 在 fallback 场景下返回 `use_gnn=False`；
  - `BranchingPolicyService` 端到端输出 shape 与 id 还原正确性。

---

## 8. 部署与发布

- 目前**没有自动化部署流程**（无 CI/CD、Dockerfile、GitHub Actions、Makefile）。
- 项目以**可导入 Python 包**形式存在，但缺少 `pyproject.toml` / `setup.py`，不能直接 `pip install -e .`。
- 若需发布/安装，建议先补全 `pyproject.toml`（或 `setup.py`），声明 `name="omt_branching"`、版本 `0.1.0`（已在 `__init__.py` 中定义）、依赖 `torch>=2.0`、`numpy>=1.21`。
- 运行环境需保证 CUDA/CPU 版 PyTorch 与硬件一致；`requirements.txt` 未区分 CPU/GPU。

---

## 9. 安全注意事项

- 当前代码不涉及网络通信、文件反序列化（除 PyTorch 模型加载习惯外）或外部命令执行。
- `SolverSnapshot` 仅由普通 Python 标量、列表、字典和 dataclass 组成，接收外部输入时相对安全；但仍应校验 id 唯一性、数值范围等，避免构造异常图导致 `index_add_` 越界。
- 若未来引入 `torch.load()` 加载训练好的策略权重，请注意 PyTorch 默认使用 pickle，可能执行任意代码；建议仅加载可信来源的模型，并考虑使用 `weights_only=True`（PyTorch 2.0+）。
- 项目没有 secret、token 或环境变量配置，无需 `.env` 管理。

---

## 10. 实验脚本

新增 `experiments/` 目录用于离线验证：

```text
experiments/
├── synthetic_omt.py        # 合成 OMT 快照生成器
├── oracle.py               # 专家标签 / 强 heuristic baseline
├── train_eval.py           # 训练与评估指标
├── run_experiments.py      # 多实验主脚本（支持 python -m experiments.run_experiments [exp_name]）
├── run_quick_experiments.py# 快速验证脚本
├── solver_sim.py           # 简化 OMT 求解器模拟器
├── sim_compare.py          # GNN / VSIDS / Oracle 模拟对比
├── profile_inference.py    # 推理开销随规模变化
├── compile_speedup.py      # torch.compile 加速测试
├── reinforce_online.py     # 在线 REINFORCE 微调框架
├── analyze_results.py      # 读取 JSON 结果并打印表格
├── plot_results.py         # 训练曲线与对比柱状图
└── EXPERIMENTS.md          # 实验报告
```

运行完整实验：

```bash
conda activate omt
python -m experiments.run_experiments
```

主要结论（合成数据）：GNN top-1 0.43–0.73，显著高于 VSIDS 0.067；deep_gnn（4 层）表现最佳；跨分布泛化仍有提升空间。

## 11. 给 AI 助手的快速检查清单

修改代码前请确认：

1. 是否同步更新了 `FeatureSpec` 中对应的节点/边维度？
2. 是否同时更新了 `interfaces.py` 中的枚举与 `EDGE_SCHEMA`？
3. 是否保持 docstring 与注释的中文风格？
4. 是否通过 `python -m examples.demo` 验证端到端流程？
5. 是否在 `__init__.py` 中导出新增公共 API？

---

*本文件基于 `README.md`、`plan.md` 以及 `omt_branching/`、`examples/` 下全部源码生成，未添加项目未体现的假设。*
