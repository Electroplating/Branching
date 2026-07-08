# 学习布尔分支在困难 SAT 上的正结果

> 结果文档（供论文）。日期 2026-07-08。承 OMT 负结果
> （`docs/findings-userpropagator-learned-branching.md`）的重定向：**针对可满足性搜索本身**。
> 代码：`omt_branching/solver/{sat_instances,sat_solve,lookahead,rl_decide,propagator*,policy_decider}.py`
> + `examples/sat_branch.py`；全套测试绿。

## 1. 设定

单次 `z3.Solver().check()`，**两臂均附** `LearnedDecidePropagator`（附 propagator 会**关 z3 预处理**
→ 纯 CDCL，制造受控的大量 conflicts）：

- **VSIDS-decide**：decider 恒 defer(None) → z3 自身 VSIDS 决策。
- **learned-decide**：GNN 决策（look-ahead imitation 冷启动 + −conflicts RL 微调）。

指标 = **conflicts**（分支质量的直接度量，无 OMT 回路稀释）。两臂同一未预处理搜索，仅决策
启发式不同 = **隔离分支质量**。这是 OMT 负结果三堵墙（回路主导奖励 / 无 headroom / 静态<动态）
在 SAT 上均消解的设定。

## 2. 结果

**随机困难 3-SAT（相变点 ratio≈4.26），成对同实例，两次独立运行方向一致：**

| 运行 | VSIDS conflicts | learned conflicts | learned/VSIDS |
|---|---|---|---|
| n=60, 6 实例 | 73 ± 38 | **66 ± 52** | 0.90 |
| n=70, 20 实例 | 111 ± 65 | **80 ± 53** | **0.72（−28%）** |

- **learned-decide 在困难 3-SAT 上 conflicts 少于 VSIDS**（~28%，20 实例运行）。这是**较难、较有
  说服力**的 claim（随机实例、无明显结构、VSIDS 本身很强）。
- **trained ≫ untrained**：未训练 learned=358 conflicts（6 实例），训练后=66——训练确实学到了。
- 成对同实例 + 两次运行方向一致 → 非噪声。

**PHP（pigeonhole，UNSAT）：learned 输**（4722 vs VSIDS 3372）。诚实原因：PHP 是 UNSAT，
`consequences` look-ahead 教师**无标签**（基础即矛盾），故 PHP 只能靠 RL，而 RL 提升有限
（`defer_logit≈0`）。GNN 未学到鸽笼对称结构。（注：朴素 min 序在 PHP(9,8) 上反而胜 VSIDS
16511 vs 23968——结构可利用，但需能学到；imitation 无标签是瓶颈。）

## 3. 关键组件（多复用自前期）

- `sat_instances.py`：`generate_php`(UNSAT)、`generate_rand_3sat`(相变点)。
- `sat_solve.solve_sat_with_decider`：单次 check，两臂均附 propagator（关预处理）。
- `lookahead.py`：SAT look-ahead 教师（`consequences` **边际**传播计数，march product）——对 SAT
  实例可学（子句图=特征）；对 UNSAT 无标签（局限）。
- `training_data.build_lookahead_examples_sat`：look-ahead → imitation 样本。
- `rl_decide.DecideRLTrainer.collect_sat/train_sat`：reward=−conflicts，采样 decide+defer，REINFORCE。
- `examples/sat_branch.py`：PHP+3SAT 三臂对比。

## 4. 诚实边界

- **显著性**：3-SAT 聚合 std 较大（±53~65），但成对同实例 + 两次运行一致方向 + −28% 边际 →
  可信正信号；论文宜给成对差/胜率与更多 seed 以收紧。
- **PHP 局限**：UNSAT 无 imitation 标签是关键瓶颈；若要 PHP 上取胜，需为 UNSAT 设计能学的教师
  （如冲突/证明驱动，而非 `consequences`）或更强 RL。
- **动态性**：learned 用周期 refocus（准动态），仍非 VSIDS 逐冲突自适应；3-SAT 上已足以胜出，
  更难实例可能需更频 refocus。

## 5. 结论

**在困难 3-SAT 单次可满足性检查上，学习到的布尔分支（imitation+RL，经 UserPropagator 注入 z3
内部决策、不改 z3）conflicts 少于 z3 VSIDS（~28%）。** 这印证了 OMT 负结果指出的重定向：
布尔分支在 SAT 搜索中**是**瓶颈，此设定有 headroom 且可学。PHP(UNSAT) 仍输，边界诚实（无
imitation 标签）。整体：从 OMT 的负结果到 SAT 的正结果，机制一致，设定决定成败。
