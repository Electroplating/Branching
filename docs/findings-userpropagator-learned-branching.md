# 用 UserPropagator 学习 z3 内部布尔分支：框架与诚实的负结果

> 结果文档（供论文"负结果 / 分析"章节）。日期 2026-07-08。全部代码/测试在
> `omt_branching/solver/{propagator,propagator_snapshot,policy_decider,decide_omt,lookahead,rl_decide}.py`
> 与 `examples/decide_branch.py`；116 tests 全绿。可行性 spike：`docs/ref/spike_userpropagator_decide.py`。

## 1. 目标与动机

目标 **(a)**：用学习到的 GNN 策略**改进 z3 内部 CDCL 布尔文字/子句分支决策**（VSIDS 的替代/辅助），
**不修改 z3**。此前的 GOMT 外层 F-Split / LIA B&B 是**外层**分支，够不到 z3 内部决策。

## 2. 机制（有效，已证）

z3 4.15.4 的 `UserPropagateBase.add_decide` + `next_split` 允许**外部接管 z3 内部布尔决策**，
无需改 z3。**已在真实 SAT 实例上验证**（spike）：

- decide 回调被触发、`next_split` 被 z3 接受；
- 正确性不变（三种决策策略结果一致 `sat`）；
- 两种策略在同一实例上 rlimit 不同（4694 vs 7880）——**确实控制了内部分支**。

**关键约束（已验证）**：`z3.Optimize` **不支持** propagator（`Solver`-only）。故 OMT 学习分支
必须走 **Solver 线性搜索回路**（Solve + Better-cut，直到 UNSAT）；GOMT 骨架 + 内部 decide 两层统一。

## 3. 实现（三阶段，全部落地并测试）

- **Phase 1（管道）**：`LearnedDecidePropagator`（接管决策，decider 返回 None 则退回 VSIDS）+
  `build_bool_snapshot`（原子 + 子句共现图 + 度/极性特征）+ `PolicyDecider`（GNN 优先级 + 周期
  refocus）+ `solve_omt_with_decider`（三臂：native / VSIDS-decide / learned-decide）。
  **结果**：learned 臂 `== native`（管道正确），propagator 生效（~362 决策/实例），可测量。
- **Phase 2a（imitation）**：SAT look-ahead 教师（`consequences` 计**边际**传播强度，march 风格
  product）→ `ImitationTrainer`。**可学**（branch 损失 3.27→冻结…经三处根因修复后 2.92→2.86 下降）。
- **Phase 2b（RL）**：`SamplingPolicyDecider`（对 `[defer, 原子分数]` softmax 采样，defer→退回
  VSIDS）+ `DecideRLTrainer`（reward=−log1p(rlimit)，per-instance baseline，REINFORCE）。

## 4. 负结果（核心）

**训练后的 learned-decide 在这些实例上无法胜过 VSIDS-decide**，且原因是**结构性**的、已量化诊断。

代表数据（更难布尔结构整数 OMT，20 测试，imitation 40 + RL 2 iter；三臂 `== native`，match=1）：

| 臂 | rlimit | conflicts |
|---|---|---|
| native(z3 Optimize) | 1,856,813,933 | — |
| VSIDS-decide | 1,856,848,557 | **6.0** |
| learned-decide（trained） | 1,856,863,580 | **26.4** |

`defer_logit` 训练后 ≈ 0.014。

## 5. 三堵墙（分析）

1. **奖励被 OMT 回路开销主导 → 布尔分支不是瓶颈。** rlimit 三臂几乎相同（~1.857B），因为它由
   OMT 线性搜索回路（数百次 Better-cut 迭代，每次一次完整 solve）主导，而非布尔分支。conflicts
   有别（6 vs 26）却只是总量的舍入误差。**即便分支完美，也几乎不动总代价**——在 Solver-loop OMT 下，
   瓶颈是回路而非布尔分支。这也使 −rlimit 作 RL 奖励**无信号** → `defer_logit≈0`，RL 不学。
2. **无 headroom。** 即使"更难"实例，VSIDS 仍仅 ~6 conflicts。VSIDS 在这些实例上已近最优，任何学习
   先验都无从改进（同 finding ⑤ 的形态：无法胜过一个在该实例上已近最优的基线）。
3. **静态学习覆盖 < 动态 VSIDS。** VSIDS 从冲突中**动态**调整变量活跃度；我们经周期 refocus 得到的
   是**静态**（图结构）优先级 + 硬覆盖每个决策，丢弃了 VSIDS 的冲突驱动自适应。CLAUDE.md 本意是把
   GNN 先验**混入** VSIDS 活跃度（软偏置，保留动态），但 z3 propagator 只暴露硬 `next_split`，
   不暴露活跃度注入 API。

## 6. 可复用资产（不论结果）

- **机制**：可外部操控 z3 内部布尔决策（`UserPropagator`），达 `== native`，可测量——这是可复用框架。
- **look-ahead 教师**（`consequences` 边际传播）、`build_bool_snapshot`（子句图 + 结构特征）、
  三臂 harness、采样 decider + defer 动作 + REINFORCE 训练器——均可迁移到新设置。
- **经验教训**：标签必须能被 GNN 特征预测（子句图↔传播同构才可学；LIA 分离度需缺失的 LP 特征则不可学）。

## 7. 指向的重定向（证据支撑，非猜测）

要让**学习布尔分支真正有意义**，应针对**可满足性搜索本身**——单次困难 SMT/SAT `check()`
（如 pigeonhole、相变点附近的困难随机 3-SAT、或人造 SMT），此时布尔搜索**就是**瓶颈、VSIDS 会探索
成千上万次冲突（有 headroom），奖励用 **−conflicts**（而非被回路主导的 rlimit）。这正是本负结果
提供的**证据**：布尔分支在 OMT 回路下不是瓶颈，故须换到布尔搜索主导的设置。

## 8. 结论

- **机制成立**：不改 z3 即可学习/操控其内部布尔分支。
- **在 OMT（Solver-loop）上，外部硬覆盖无法胜过 VSIDS**——三堵墙（回路主导奖励、无 headroom、
  静态 vs 动态）均已量化。
- **诚实贡献**：一个可操控 z3 内部分支的框架 + 对"为何外部硬覆盖难胜 VSIDS"的严格分析 + 明确的
  重定向（针对可满足性搜索）。
