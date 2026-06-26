# 设计文档：z3 ↔ Neural GOMT 桥接 API（基于 GOMT calculus）

- 日期：2026-06-26
- 状态：已与用户确认架构，待实现
- 关联：`GOMT.pdf`（Tsiskaridze, Barrett, Tinelli, *Generalized Optimization Modulo Theories*）、`plan.md`、`omt_branching/` 现有 Neural 组件

---

## 1. 目标与范围

实现一套**完整、准确、清晰、且不增冗余实体**的标准 API 接口，把外部 **OMT 求解器（z3）** 与现有 **Neural 分支策略组件**（`omt_branching/`）联通起来。

整套桥接以 **GOMT calculus**（`GOMT.pdf` 第 4 节）为理论骨架，在 Python 中实现该 calculus 的可执行版本，并把 Neural 策略**精确地插入到 calculus 唯一的启发式自由点 —— F-Split 规则**。z3 作为 calculus 的 `Solve` / `Optimize` 后端（理论 oracle），**不修改 z3 源码、不重编译**。

### 1.1 v1 范围（YAGNI）

- 单目标 **最小化 / 最大化**，理论为 **LIA / LRA**（线性整数 / 实数算术）。与 `plan.md` 优先级、`ObjectiveInfo.sense_is_min`、现有实验一致。
- `Better` / 序 `≺` 设计为**可插拔**，为后续 lexicographic / Pareto 留接口，但 v1 **不实现**多目标。
- 整数 / 数值分支 = **域切分**（`x ≤ m` ∨ `x ≥ m+1`，切分点取自 incumbent 值或域中点）。z3 公开 API 不暴露 LP 松弛的分数值，因此**不伪造** `lp_value` / `reduced_cost` 等字段。

### 1.2 明确的非目标

- 不修改 z3 C++ 源码（用户的"尽量少损害 z3"约束）。
- 不通过 user-propagator `decide` 回调做原生 in-place 控制（option 2，已排除）。
- 不实现多目标（lexicographic / Pareto / MaxSMT）的 v1 求解，仅保留扩展点。
- 不读取 z3 内部 VSIDS / simplex basis / pseudocost（公开 API 无法获取）。

---

## 2. GOMT calculus 回顾（实现依据）

状态 `Ψ = ⟨I, Δ, τ⟩`：
- `I`：迄今找到的最优解（interpretation / model）。
- `Δ`：描述"仍可能存在更优解"的剩余搜索空间的公式。
- `τ`：把 `Δ` 划分成若干 branch 的公式序列；不变式 `φ ⊨ (⋁ τ_i ⇔ Δ)`。

基元：
- `Solve(F)`：SMT 判定，SAT 返回 model，UNSAT 返回 `⊥`。
- `Better(I)`：公式，满足 `I' ⊨ Better(I)` 当且仅当 `I' <_GO I`。最小化且 `t^I = c` 时 `Better(I) = (t < c)`；最大化时 `(t > c)`。
- `Optimize(F, t)`（**更强的 Solve**，§4.2 Hybrid search / Thm 4）：直接返回 branch 内对目标 `t` 最优的 model。由 z3 `Optimize` 实现。

初始状态：`I₀ = Solve(φ)`，`Δ₀ = Better(I₀)`，`τ₀ = (Δ₀)`（假设 `φ` 可满足）。

派生规则（Fig. 1）：

| 规则 | 前提 | 结论 |
|---|---|---|
| **F-Split** | `τ≠∅`, `ψ=Top(τ)`, `φ⊨ψ⇔⋁_{j=1}^k ψ_j`, `k≥1` | `τ := (ψ₁,…,ψ_k) ∘ Pop(τ)` |
| **F-Sat** | `τ≠∅`, `ψ=Top(τ)`, `Solve(φ∧ψ)=I'≠⊥`, `Δ'=Δ∧Better(I')` | `I:=I'`, `Δ:=Δ'`, `τ:=(Δ')` |
| **F-Close** | `τ≠∅`, `ψ=Top(τ)`, `Solve(φ∧ψ)=⊥` | `Δ:=Δ∧¬ψ`, `τ:=Pop(τ)` |

**饱和**（`τ=∅`）⇒ `I` 为最优解（Theorem 1，Solution Soundness）。

**关键洞察**：F-Split 的 `ψ₁,…,ψ_k` 如何选择，calculus **不指定**——这正是 branching 启发式的自由度，也正是 Neural 策略的插入点。F-Sat / F-Close 是机械步骤。无论 Neural 如何选择 split，calculus 的 soundness 由构造保证。

---

## 3. 架构

新增子包 `omt_branching/solver/`，**不改动** Neural 侧（`input/` `model/` `output/`），**z3 完全隔离**在单一模块。

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
        │Neural | Base   │   │  Z3Backend      │  ← 唯一 import z3 的文件
        │ Strategy       │   │ solve/optimize  │
        └──────┬─────────┘   └─────────────────┘
   F-Split ⇒   │  三个桥接缝 = "标准 API"：
        (a) Z3SnapshotExtractor : z3 state → SolverSnapshot
        (b) BranchingPolicyService.advise : Snapshot → Advice   （现有，不改）
        (c) AdviceToSplit : Advice → (ψ₁,ψ₂) z3 公式
```

### 3.1 两个核心 Protocol 接口（解耦关键）

**`SolveBackend`（Protocol）** —— OMT 后端契约，calculus 通过它调用理论求解：
```python
class SolveBackend(Protocol):
    def solve(self, constraint) -> Optional[Model]: ...
    def optimize(self, constraint, objective, sense) -> Optional[tuple[Model, Value]]: ...
    def value(self, model: Model, term) -> Value: ...
    def is_true(self, model: Model, atom) -> bool: ...
```
z3 是其唯一实现（`Z3Backend`）。**除 `z3_backend.py` 外没有任何文件 import z3。**

**`BranchingStrategy`（Protocol）** —— calculus 咨询的决策者：
```python
class BranchingStrategy(Protocol):
    def propose(self, state: GOMTState, backend: SolveBackend) -> SplitDecision: ...
```
`NeuralStrategy` 与 `BaselineStrategy`（二分搜索）都实现它。**calculus 永远看不到神经网络；strategy 永远看不到 z3 内部（只经 backend 接口）。**

`SplitDecision`：描述对 `Top(τ)` 的处理——给出有序的 `ψ₁,…,ψ_k`（F-Split 用），或指示直接走 F-Sat（linear 模式 / leaf）。

### 3.2 三个桥接缝（用户要的"联通 OMT 与 Neural 的 API"）

1. **(a) `Z3SnapshotExtractor`：z3 state → `SolverSnapshot`**（输入半边）
   遍历 `φ`、`Δ∧ψ` 的 z3 AST，枚举：
   - `bool_vars`：布尔结构变量 + 理论原子对应的抽象布尔变量。
   - `theory_atoms`：线性原子 → `var_coeffs` / `rhs` / `kind`（`<=`/`<`/`=` 等）。
   - `numeric_vars`：数值变量（`is_integer`、`lower_bound`/`upper_bound` 来自 `Δ`/`φ` 中已累积的界、当前 model 值）。
   - `clauses`：CNF 结构（来自 `φ` 的子句视图，best-effort）。
   - `objective`：`var_coeffs`（来自 `t`）、`sense_is_min`、`incumbent`（当前 `I` 的目标值）、`best_bound`。
   - `search_state`：`depth`/`decision_level`（来自派生长度、`|τ|`）、`search_mode`、`conflict_count`（F-Close 次数）等。
   - `candidate_bool_ids` / `candidate_numeric_ids`：当前 branch 下可分支对象。
   **诚实原则**：结构字段如实填，深层 theory-internal 字段（per-var VSIDS、`reduced_cost`、`is_basic`、`pseudocost_*`）留在契约的"缺省/None"，由 graph builder 的 present-flag 编码处理。API.md 用一张表写明"z3 能/不能提供什么"。

2. **(b) `BranchingPolicyService.advise`：`SolverSnapshot` → `BranchingAdvice`**（现有，零改动）

3. **(c) `AdviceToSplit`：`BranchingAdvice` → `SplitDecision`**（输出半边）
   - 取 `advice.top_candidate()` / 候选排序 → 选择分支原子或数值变量。
   - 布尔原子分支：`ψ₁ = ψ ∧ atom`，`ψ₂ = ψ ∧ ¬atom`；`advice.phase` 决定先探哪个（顺序即 `τ` 中位置，§4.2 Search order）。
   - 数值/整数分支：选变量 + 切分点 `m` → `ψ₁ = ψ ∧ (x ≤ m)`，`ψ₂ = ψ ∧ (x ≥ m+1)`；`integer_split` 方向决定先后。
   - `advice.fallback is True`（超时/低置信/OOD）→ 退回 `BaselineStrategy` 的默认切分，保证不卡死。
   - 满足 F-Split 前提 `φ⊨ψ⇔ψ₁∨ψ₂`：因 `ψ₂=ψ∧¬atom`、`ψ₁=ψ∧atom`，二者析取 `=ψ∧(atom∨¬atom)=ψ`，**构造上即满足**。

### 3.3 F-Sat 的两种 oracle 模式（用户的"更强 Solve"）

- **`plain`（默认，研究模式）**：F-Sat 用 `Solve`；Neural **F-Split 驱动搜索**，策略真正掌控 branching，可度量。
- **`hybrid`**：F-Sat 用 z3 `Optimize` 作 leaf 加速器（§4.2 / Thm 4）。纯 `hybrid` 且不 split = z3 原生 OMT，作为 **baseline** 对照。

---

## 4. 模块清单（`omt_branching/solver/`）

| 文件 | 职责 | 依赖 |
|---|---|---|
| `interfaces.py` | `SolveBackend`、`BranchingStrategy` Protocol；`SplitDecision`、`GOMTState`、`SolveResult` 等数据类 | 标准库 |
| `problem.py` | `GOMTProblem ⟨t, ≺, φ⟩`；`better(value) -> formula`；`sense` | backend 抽象 |
| `calculus.py` | `GOMTSolver`：状态机，F-Split/F-Sat/F-Close 主循环，统计 | interfaces, problem |
| `z3_backend.py` | `Z3Backend`（唯一 import z3）：`solve`/`optimize`/`value`；构造 z3 表达式 | z3 |
| `extractor.py` | `Z3SnapshotExtractor`：z3 state → `SolverSnapshot` | z3, input/ |
| `strategy.py` | `NeuralStrategy`（含 `AdviceToSplit`）、`BaselineStrategy`（二分） | output/, model/ |
| `bridge.py` | `NeuralGOMTSolver` 门面：组装 problem+backend+extractor+strategy+service，`solve()` 返回 (model, value, stats) | 全部 |
| `__init__.py` | 公共导出 | — |

文档与示例：
- `omt_branching/solver/API.md`：接口文档（中文）。含两个 Protocol、三个缝、GOMT⇄契约映射表、"z3 能/不能提供"表、最小集成示例。
- `examples/z3_demo.py`：真实 OMT(LIA) 实例，Neural-guided GOMT vs z3-native，打印 stats。
- 测试见 §6。

---

## 5. 数据流（一次完整 solve）

```
GOMTProblem(φ, t, sense)
    │  Z3Backend.solve(φ) = I₀ ; Δ₀=Better(I₀) ; τ₀=(Δ₀)
    ▼
GOMTSolver 循环，直到 τ=∅：
    ψ = Top(τ)
    decision = BranchingStrategy.propose(state, backend)
        └─(Neural) extractor → SolverSnapshot → service.advise → AdviceToSplit → SplitDecision
    if decision = SPLIT(ψ₁,ψ₂):  F-Split → τ=(ψ₁,ψ₂)∘Pop(τ)
    else:                         # F-Sat / F-Close 由 Solve 结果决定
        r = backend.solve(φ∧ψ)   (或 optimize, hybrid 模式)
        if r≠⊥:  F-Sat  → I=r, Δ=Δ∧Better(r), τ=(Δ)
        else:    F-Close→ Δ=Δ∧¬ψ, τ=Pop(τ)
    ▼
τ=∅ ⇒ 返回 (I, value(I,t), stats{splits, sats, closes, solve_calls, wall_ms})
```

---

## 6. 测试与验证策略（"用 z3 来 debug"）

无现有测试框架；本桥接正确性关键，新增 `tests/`（pytest 或纯脚本）：

1. **Oracle 一致性（核心）**：随机生成 N 个 LIA 单目标实例，Neural-guided `GOMTSolver` 返回的最优值必须 == z3-native `Optimize` 的最优值。这是 calculus 实现正确性的金标准。即便策略未训练（随机权重），最优值也必须一致（soundness 与策略无关，Thm 1）。
2. **不变式断言**：循环中可选开启 `φ ⊨ (⋁τ_i ⇔ Δ)` 的抽样检查（用 backend 验 `Solve(φ ∧ ¬(⋁τ_i ⇔ Δ))=⊥`）。
3. **F-Close / fallback**：构造 UNSAT branch，验证走 F-Close；构造 `advice.fallback=True`，验证退回 baseline 且仍得最优。
4. **Extractor 健壮性**：缺失字段、空 branch、单原子公式不崩。
5. **端到端**：`examples/z3_demo.py` 跑通，stats 合理（splits>0 表示 Neural 真在分支）。

---

## 7. 与现有约定一致性

- 文档/注释中文；`__all__`、类型注解英文标识符（AGENTS.md §6）。
- 配置即 dataclass：`GOMTConfig`、`Z3BackendConfig` 等 frozen/普通 dataclass。
- `from __future__ import annotations`；内置泛型。
- 新增公共 API 在 `omt_branching/solver/__init__.py` 导出；顶层 `omt_branching/__init__.py` 可选再导出 `NeuralGOMTSolver`。
- 不引入 PyG 等新依赖；新增运行期依赖仅 `z3-solver==4.15.4.0`（写入 `requirements.txt`，与系统 `/usr/local/bin/z3` 4.15.4 同版本，互不影响）。

---

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| z3 AST → 线性原子解析覆盖不全（非线性、复杂嵌套） | v1 限定 LIA/LRA 线性原子；非支持结构在 extractor 中跳过并 `log`，不静默 |
| F-Sat=plain 时 Neural 不收敛 / Zeno 链 | 设置最大派生步数 / 时间预算；超限按 anytime（Thm 后注）返回当前 `I`，并标记非最优 |
| `Better` 浮点比较精度（LRA） | 数值容差 `eps`；整数走精确比较 |
| 策略未训练导致分支差 | 正确性不依赖策略；性能对比单列。demo 可先跑几步 imitation 训练 |
| z3 版本/绑定不一致 | 固定 `z3-solver==4.15.4.0`，装入 `omt` conda env，不动系统 z3 |

---

## 9. 验收标准

1. `omt_branching/solver/` 七个模块齐备，仅 `z3_backend.py`/`extractor.py` import z3。
2. `API.md` 完整描述两个 Protocol、三个缝、映射表、能力边界。
3. Oracle 一致性测试在 N≥20 随机 LIA 实例上全部通过（最优值与 z3-native 一致）。
4. `python -m examples.z3_demo` 在 `omt` env 跑通，输出 Neural-guided 与 baseline 的 stats 对比，`splits>0`。
5. 不修改 Neural 侧任何现有文件的行为；不触碰系统 z3。
