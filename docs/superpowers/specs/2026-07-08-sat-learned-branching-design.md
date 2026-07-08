# 学习布尔分支：困难 SAT 可满足性（pivot）—— 设计

## 0. 背景与转向依据

OMT 负结果（`docs/findings-userpropagator-learned-branching.md`）显示三堵墙：rlimit 被 OMT 回路
主导 / 无 headroom / 静态覆盖<动态 VSIDS。**pivot 到单次困难 SAT 可满足性检查**，三堵墙均消解
（已用 probe 验证）：

- **附加 propagator 会关闭 z3 预处理 → 纯 CDCL 搜索**：PHP(9,8) 无 propagator conflicts=0，
  附 propagator 后 VSIDS-decide=**23968** conflicts、朴素固定序=**16511**——headroom 巨大，且
  **朴素静态序已胜 VSIDS**（结构实例上 VSIDS 通用启发式盲于对称性）。这正是 learn-to-branch 有效区。
- 奖励 = **−conflicts**（分支直接指标，无 OMT 回路稀释）。

## 1. 目标与成功标准

- **成功**：trained learned-decide 的 **conflicts < VSIDS-decide**（两臂**均附 propagator**，同一
  未预处理搜索，仅决策启发式不同 = 隔离分支质量）。多 seed mean±std。
- **两个claim**：(a) **PHP**：learned < VSIDS 且 **trained ≫ untrained**（证明 GNN *学到*结构，
  非"任意静态序皆胜"）；(b) **随机 3-SAT**（相变点）：learned < VSIDS（更难、更有说服力；若平/负
  如实报告，PHP 结论仍立）。
- 不改 z3；复用 Phase 1/2 全部管道。

## 2. 架构

```
单次 z3.Solver().check(assertions)，附 LearnedDecidePropagator（关预处理→纯 CDCL）
  两臂均附 propagator：
    VSIDS-decide  = decider 恒 defer(None) -> z3 自身 VSIDS 决策
    learned-decide = GNN decider
  指标 = conflicts（大、完全受控）
```

## 3. 组件

### 3.1 SAT 实例生成（`omt_branching/solver/sat_instances.py`，新）

- `generate_php(m) -> tuple[list, list]`：PHP(m+1, m)——m+1 鸽 m 洞，**UNSAT**；返回 (atoms, clauses)。
  子句：每鸽入某洞 `Or(p[i][*])`；无两鸽同洞 `Or(¬p[i1][j], ¬p[i2][j])`。
- `generate_rand_3sat(n, ratio=4.26, seed=0) -> tuple[list, list]`：相变点附近随机 3-SAT。
- 轻量返回 (atoms: list[z3.BoolRef], clauses: list[z3.BoolRef])。

### 3.2 SAT 求解 harness（`omt_branching/solver/sat_solve.py`，新）

```python
def solve_sat_with_decider(assertions, atoms, decider_factory=None) -> dict:
    # 恒附 propagator（两臂都关预处理，公平）。decider_factory=None -> defer-always(=VSIDS 臂)。
    # 返回 {result:"sat"/"unsat", conflicts, decisions, rlimit}
```

### 3.3 −conflicts RL（`rl_decide.py` 扩展）

- `DecideRLTrainer.collect_sat(assertions, atoms)`：用 `solve_sat_with_decider` + `SamplingPolicyDecider`
  跑一次，reward = **−conflicts**（可 log1p 压缩），返回 (steps, reward, res)。复用现有 `update()`
  （solve-agnostic：takes steps, reward, key）与 per-instance baseline。
- `train_sat(instances: list[(assertions, atoms)], iterations)`。

### 3.4 look-ahead imitation for SAT（`training_data.py` 扩展）

- `build_lookahead_examples_sat(problems, config)`：`problems = list[(assertions, atoms)]`；每个用
  `build_bool_snapshot(assertions)` 建图、`lookahead_scores(assertions, atoms)` 打标签，映射到
  BOOL_VAR 局部索引 → `RankingExample`。复用 `ImitationTrainer`。

### 3.5 实验（`examples/sat_branch.py`，新）

生成 PHP + 3-SAT → imitation 冷启动 → RL(−conflicts) 微调 → 三臂对比（VSIDS-decide vs
learned-decide 的 conflicts，两臂均附 propagator），多 seed 聚合 mean±std。断言 result 一致
（sat/unsat 与 z3 default 一致，正确性）。

## 4. 数据流

```
PHP/3SAT (atoms, clauses) → build_bool_snapshot(子句图) → look-ahead 标签 → imitation
   → RL(采样 decide + defer, reward=−conflicts) → trained policy → solve_sat_with_decider
   → 三臂 conflicts 对比 vs VSIDS
```

## 5. 关键风险

- **随机 3-SAT 可能仍难胜 VSIDS**：如实报告（PHP claim 仍立）。
- **PHP 胜 VSIDS 部分"免费"**（对称结构）：故须验证 trained ≫ untrained（GNN 确实学到），否则只是
  "任意静态序皆胜"。
- **refocus 频率 vs 动态性**：refocus 越频，GNN 越能随赋值自适应（近 VSIDS 动态），但越慢——实验调。
- **UNSAT 实例**：PHP 是 UNSAT，全程冲突驱动证明；conflicts 指标良好定义。

## 6. 复用 vs 新建

- **复用**：`LearnedDecidePropagator`、`build_bool_snapshot`（子句图+结构特征）、`SamplingPolicyDecider`
  +defer、`DecideRLTrainer.update()`/baseline、`lookahead_scores`、GNN、三臂 harness 思路。
- **新建**：`sat_instances.py`、`sat_solve.py`、`collect_sat`/`train_sat`、`build_lookahead_examples_sat`、
  `examples/sat_branch.py`。

## 7. 非目标

- 不改 z3；v1 用纯 SAT（PHP + 3-SAT）；SMT/theory-atom 扩展为后续（原子机制相同）。
- 不与 z3-default（预处理后 conflicts=0）比——非同类；只比 propagator 下 VSIDS vs learned。
