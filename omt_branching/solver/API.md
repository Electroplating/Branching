# z3 ↔ Neural GOMT 桥接 API

本文件描述把 **OMT 求解器（z3）** 与 **Neural 分支策略组件** 联通的标准接口。
桥接以 **GOMT calculus**（`GOMT.pdf` 第 4 节，Tsiskaridze, Barrett, Tinelli，
*Generalized Optimization Modulo Theories*）为理论骨架：在 Python 中实现该 calculus
的可执行版本，把 Neural 策略精确插入到 calculus **唯一的启发式自由点 —— F-Split**。

- z3 **不被修改、不重编译**；仅用 pip 的 `z3-solver`（与系统 `/usr/local/bin/z3`
  同版本，互不影响）。
- 现有 Neural 侧（`input/` `model/` `output/`）**零改动**，原样复用
  `SolverSnapshot` / `BranchingAdvice` / `BranchingPolicyService`。

---

## 1. 一眼看懂：数据流与三个桥接缝

```
                 GOMTProblem ⟨t, ≺, φ⟩        (problem.py)
                        │
            ┌───────────▼─────────────┐
            │  GOMTSolver (calculus.py) │  theory-agnostic 引擎
            │  Ψ=⟨I,Δ,τ⟩  F-Split/Sat/Close
            └──┬───────────────────┬───┘
   consults    │                   │  delegates
 BranchingStrategy             SolveBackend
        ┌──────▼────────┐   ┌──────▼─────────┐
        │Neural | Base   │   │  Z3Backend      │  ← 唯一封装 z3 的求解后端
        │ Strategy       │   │ solve/optimize  │
        └──────┬─────────┘   └─────────────────┘
   F-Split ⇒   │  三个桥接缝（标准 API）：
        (a) Z3SnapshotExtractor : z3 state → SolverSnapshot
        (b) BranchingPolicyService.advise : Snapshot → Advice   （现有，不改）
        (c) AdviceToSplit（NeuralStrategy 内） : Advice → (ψ₁,ψ₂) z3 公式
```

z3 依赖被限制在 **adapter 层**：`z3_backend.py`（求解/公式代数）与 `extractor.py`
（AST 抽取）。`interfaces.py` / `problem.py` / `calculus.py` / `strategy.py` 不直接
`import z3`，只经 `SolveBackend` 操作不透明句柄（`Model`/`Term`/`Constraint`/`Atom`）。

---

## 2. 两个核心 Protocol 接口

### 2.1 `SolveBackend`（`interfaces.py`）

OMT 后端契约：calculus 与策略经它做理论求解与公式代数。z3 实现为 `Z3Backend`。

| 方法 | 语义 |
|---|---|
| `solve(constraint) -> Model \| None` | SMT 判定；SAT 返回 model，UNSAT 返回 `None`。 |
| `optimize(constraint, objective, sense) -> (Model, Value) \| None` | **更强的 Solve**：返回 branch 内对 `objective` 最优的解（z3 `Optimize`）。 |
| `value(model, term) -> Value` | 求值（`int` / `Fraction`）。 |
| `is_true(model, atom) -> bool` | `atom` 在 `model` 下是否为真。 |
| `conjoin(*constraints) -> Constraint` | 合取；零参数为逻辑真。 |
| `negate(constraint) -> Constraint` | 取非。 |
| `better(objective, value, sense) -> Constraint` | `Better(I)`：MIN→`objective<value`，MAX→`objective>value`。 |
| `top() -> Constraint` | 逻辑真常量。 |
| `le(term, bound) -> Atom` / `ge(term, bound) -> Atom` | `term≤bound` / `term≥bound`（数值域切分用）。 |

要接入 **其它求解器**（如 OptiMathSAT），只需实现这 10 个方法，无需改动 calculus
与策略。

### 2.2 `BranchingStrategy`（`interfaces.py`）

calculus 咨询的分支决策者，唯一的启发式自由点：

```python
class BranchingStrategy(Protocol):
    def propose(self, state: GOMTState, backend: SolveBackend) -> SplitDecision: ...
```

- 返回 `SplitDecision.split([ψ₁, ψ₂, ...])`：执行 F-Split，把 `Top(τ)` 换成有序子
  公式（列表首项最先探索）。
- 返回 `SplitDecision.resolve()`：不再细分，直接对当前 branch 调用 `Solve`/`Optimize`，
  由结果机械触发 F-Sat 或 F-Close。

内置实现：`NeuralStrategy`（神经驱动）与 `BaselineStrategy`（确定性二分 / linear）。

---

## 3. GOMT calculus ⇄ 现有契约映射

| GOMT calculus（`GOMT.pdf` §4） | 本实现 | Neural 契约 |
|---|---|---|
| 状态 `Ψ=⟨I, Δ, τ⟩` | `GOMTState` | `I`→`ObjectiveInfo.incumbent` |
| `Solve(F)` | `Z3Backend.solve` | — |
| `Optimize(F,t)`（§4.2 Hybrid，Thm 4） | `Z3Backend.optimize` | `f_sat_mode="hybrid"` |
| `Better(I)=t≺c` | `Z3Backend.better` | `ObjectiveInfo.sense_is_min` |
| 初始态（Def 11） | `GOMTProblem.initial_state` | — |
| **F-Split**（选 ψ₁…ψₖ） | `BranchingStrategy.propose` | **`BranchingAdvice`**（见下） |
| F-Sat / F-Close | `GOMTSolver.run` 主循环 | — |
| 饱和 ⇒ 最优（Thm 1） | `GOMTResult.optimal` | — |
| Linear / Binary 策略（§4.2） | `BaselineStrategy` / `NeuralStrategy` | `SearchMode` |

**F-Split 的 Advice 翻译（缝 c，`NeuralStrategy._advice_to_split`）**：

- 数值/整数分支（LIA 的 B&B，主路径）：`advice.integer_split.num_var_id` 选变量，
  域中点 `m` 切分 `x≤m` ∨ `x≥m+1`，`branch_up` 决定先后。
- 布尔原子分支（退回）：`advice.top_candidate()` 选原子 `a`，切分 `ψ∧a` ∨ `ψ∧¬a`，
  `advice.phase_suggestions` 决定先后。
- `advice.use_gnn is False`（规模超限/低置信/OOD）→ `resolve()`，退回 linear search。

> F-Split 前提 `φ⊨ψ⇔⋁ψⱼ` 由构造保证：`ψ₁=ψ∧a`、`ψ₂=ψ∧¬a` 的析取恒等于 `ψ`；
> 数值切分 `x≤m` ∨ `x≥m+1` 覆盖整数域。故无论 Neural 如何选择，soundness 不变。

---

## 4. z3 能/不能提供哪些 `SolverSnapshot` 字段（诚实边界）

抽取器只能用 z3 **公开** API。下表说明实际填充情况；未填字段保持契约缺省（`None`/0），
由 graph builder 的 present-flag 编码处理，不影响推理。

| 字段类别 | 能否由 z3 提供 | 来源 |
|---|---|---|
| 数值变量、`is_integer`、线性原子 `var_coeffs`/`rhs`/`kind` | ✅ | 遍历 `φ` 的 AST |
| 变量上下界 `lower/upper_bound` | ✅（来自盒约束/已下降的 split） | 单变量原子扫描 |
| 目标 `var_coeffs`/`sense_is_min`/`incumbent` | ✅ | 目标项线性分解 + 当前解求值 |
| `search_state.depth`/`decision_level`/`conflict_count` | ✅ | 派生步数 / `|τ|` / F-Close 计数 |
| 候选集合 `candidate_bool_ids`/`candidate_numeric_ids` | ✅ | 原子抽象布尔 / 全部数值变量 |
| `lp_value`（LP 松弛分数值） | ⚠️ 用 incumbent 取值替代 | z3 不暴露 LP 松弛 |
| per-var VSIDS / `lrb_score` / `chb_score` | ❌ 留缺省 | z3 公开 API 无 |
| `reduced_cost` / `is_basic`（simplex basis） | ❌ 留缺省 | z3 公开 API 无 |
| `pseudocost_up/down` | ❌ 留缺省 | z3 公开 API 无 |
| CNF `clauses` 结构 | ❌ v1 留空 | 需求解器内部子句库 |

---

## 5. 最小集成示例

```python
import z3
from omt_branching.solver import NeuralGOMTSolver, BridgeConfig, Sense, solve_native

x, y = z3.Int("x"), z3.Int("y")
hard = [x >= 0, x <= 10, y >= 0, y <= 10, x + y <= 12]
objective = 3 * x + 2 * y

# Neural 策略驱动的 GOMT 求解（默认 plain 模式，Neural F-Split 驱动搜索）
res = NeuralGOMTSolver().solve(hard, objective, Sense.MAX)
print(res.value, res.optimal, res.stats)   # 最优值 / 是否最优 / 统计

# 与 z3 原生 Optimize 对照（oracle）
assert res.value == solve_native(hard, objective, Sense.MAX)

# hybrid 模式：F-Sat 用 z3 Optimize 作 leaf 加速器
fast = NeuralGOMTSolver(config=BridgeConfig(f_sat_mode="hybrid")).solve(
    hard, objective, Sense.MAX)
```

注入**已训练**策略：把 `BranchingPolicyService(policy=trained_policy)` 传给
`NeuralGOMTSolver(service=...)`。桥接对训练与否无感（soundness 与策略质量无关）。

完整可运行示例见 `examples/z3_demo.py`：
```bash
conda run -n omt python -m examples.z3_demo
```

---

## 6. 范围与扩展点

**v1 范围**：单目标 `minimize`/`maximize`，理论 LIA/LRA，目标有界。

**扩展点**（已为之保留接口，v1 未实现）：

- **多目标 / lexicographic / Pareto**：`Sense` 与 `Z3Backend.better` 是 `≺` 的唯一落点。
  按 `GOMT.pdf` §3.2 把目标项构造成 `tup(t₁,…,tₙ)` 并用对应偏序定义 `better`，calculus
  与策略无需改动即可支持（`GOMT.pdf` Def 2–7）。
- **其它求解器后端**：实现 `SolveBackend` 的 10 个方法即可替换 z3。
- **更强的分支特征**：若目标求解器能暴露 LP 值 / reduced cost / VSIDS，在 `extractor.py`
  填入对应字段即可，无需改协议。

---

## 7. 测试与验证

`tests/solver/`（`conda run -n omt python -m pytest tests/solver -v`）：

- **`test_oracle.py`**：20 个随机 LIA 实例 × MIN/MAX，Neural-guided GOMT 最优值必须
  == z3-native `Optimize`（calculus 正确性金标准，与策略质量无关）。
- `test_z3_backend` / `test_problem` / `test_calculus` / `test_extractor` /
  `test_strategy`：各模块单元测试。
- `test_demo_smoke`：端到端运行 `examples/z3_demo.py`。
