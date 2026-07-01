"""Solver-in-the-Loop 强化学习训练示例（z3 GOMT 回路 + REINFORCE）。

流程:

1. 构造一个有界 OMT(LIA) 实例，用 z3 原生 ``Optimize`` 求出参照 optimum。
2. 训练前：用确定性 Neural 策略评估（记录 optimum / splits / solve_calls / 运行时间）。
3. 训练：在真实 z3 GOMT 回路里采集轨迹并 REINFORCE 微调
   —— 单步 reward = incumbent 提升；终局 penalty = 运行时间。
4. 训练后：再次确定性评估，与训练前对比，并断言 optimum 与 z3-native 一致（soundness）。

运行::

    python -m examples.rl_demo
"""

from __future__ import annotations

import torch
import z3

from omt_branching.model.policy import BranchingPolicy
from omt_branching.solver import (
    RLConfig,
    Sense,
    SolverInLoopRLTrainer,
    solve_native,
)


def build_instance():
    """有界 OMT(LIA) 实例：3 个整数变量 + 线性约束，最大化 3x+2y+4z。"""
    x, y, z = z3.Int("x"), z3.Int("y"), z3.Int("z")
    hard = [
        x >= 0, x <= 8,
        y >= 0, y <= 8,
        z >= 0, z <= 8,
        x + y + z <= 12,
        2 * x + y <= 14,
        y + 3 * z <= 18,
    ]
    objective = 3 * x + 2 * y + 4 * z
    return hard, objective, Sense.MAX


def _fmt(result, runtime: float) -> str:
    return (f"value={result.value} optimal={result.optimal} "
            f"splits={result.stats.get('splits')} "
            f"solve_calls={result.stats.get('solve_calls')} "
            f"runtime={runtime * 1e3:.1f}ms")


def main() -> None:
    torch.manual_seed(0)
    hard, objective, sense = build_instance()

    native_opt = solve_native(tuple(hard), objective, sense)
    print("=== Solver-in-the-Loop 强化学习示例 (OMT/LIA, maximize 3x+2y+4z) ===")
    print(f"z3-native optimum = {native_opt}\n")

    policy = BranchingPolicy()
    trainer = SolverInLoopRLTrainer(
        policy,
        RLConfig(lr=1e-3, gamma=0.98, entropy_coef=5e-3,
                 time_penalty_coef=1.0, reward_scale=0.1,
                 max_split_depth=5, max_steps=3000),
    )

    # ---------------- 训练前评估（确定性） ----------------
    before, t_before = trainer.evaluate(tuple(hard), objective, sense)
    print("训练前 (确定性策略):", _fmt(before, t_before))
    assert before.value == native_opt, "训练前 optimum 与 native 不一致！(soundness)"

    # ---------------- 强化学习训练 ----------------
    print("\n--- REINFORCE 训练（真实 z3 回路采样） ---")
    instances = [(tuple(hard), objective, sense)]
    history = trainer.train(instances, iterations=12, log=True)

    # ---------------- 训练后评估（确定性） ----------------
    after, t_after = trainer.evaluate(tuple(hard), objective, sense)
    print("\n训练后 (确定性策略):", _fmt(after, t_after))

    # ---------------- soundness & 学习信号 ----------------
    assert after.value == native_opt, "训练后 optimum 与 native 不一致！(soundness)"
    print(f"\nsoundness 校验通过：训练前后 optimum 均为 {native_opt}")

    first = history[0]["mean_return"]
    last = history[-1]["mean_return"]
    print(f"平均回报 (mean_return): 首轮 {first:.4f} -> 末轮 {last:.4f}")
    print(f"确定性 solve_calls: 训练前 {before.stats.get('solve_calls')} "
          f"-> 训练后 {after.stats.get('solve_calls')}")
    print("\n端到端 RL 训练回路验证完成。")


if __name__ == "__main__":
    main()
