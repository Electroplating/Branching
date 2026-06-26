"""z3 ↔ Neural GOMT 桥接示例：真实 OMT(LIA) 实例。

对同一实例分别用 Neural-guided GOMT、Baseline GOMT、z3-native ``Optimize`` 求解，
打印各自 optimum 与统计，并断言三者一致（GOMT soundness）。

运行::

    conda run -n omt python -m examples.z3_demo
"""

from __future__ import annotations

import z3

from omt_branching.solver import BridgeConfig, NeuralGOMTSolver, Sense, solve_native


def build_instance():
    """一个有界 OMT(LIA) 实例：3 个整数变量 + 若干线性约束，最大化线性目标。"""
    x, y, z = z3.Int("x"), z3.Int("y"), z3.Int("z")
    hard = [
        x >= 0, x <= 12,
        y >= 0, y <= 12,
        z >= 0, z <= 12,
        x + y + z <= 18,
        2 * x + y <= 20,
        y + 3 * z <= 24,
    ]
    objective = 3 * x + 2 * y + 4 * z
    return hard, objective


def main() -> None:
    hard, objective = build_instance()
    print("=== z3 ↔ Neural GOMT 桥接示例 (OMT/LIA, maximize 3x+2y+4z) ===")

    native = solve_native(tuple(hard), objective, Sense.MAX)
    print(f"z3-native   optimum = {native}")

    neural = NeuralGOMTSolver().solve(tuple(hard), objective, Sense.MAX)
    print(f"neural-GOMT optimum = {neural.value}  optimal={neural.optimal}")
    print(f"            stats   = {neural.stats}")

    baseline = NeuralGOMTSolver(config=BridgeConfig(strategy="baseline")).solve(
        tuple(hard), objective, Sense.MAX)
    print(f"baseline    optimum = {baseline.value}")
    print(f"            stats   = {baseline.stats}")

    hybrid = NeuralGOMTSolver(config=BridgeConfig(f_sat_mode="hybrid")).solve(
        tuple(hard), objective, Sense.MAX)
    print(f"hybrid      optimum = {hybrid.value}  (F-Sat 用 z3 Optimize 加速)")

    assert neural.value == native == baseline.value == hybrid.value, "optima 不一致！"
    print(f"\n一致性校验通过：所有配置 optimum 均为 {native}")
    print(f"neural 实际分支次数 splits={neural.stats['splits']}, "
          f"F-Sat={neural.stats['sats']}, F-Close={neural.stats['closes']}")


if __name__ == "__main__":
    main()
