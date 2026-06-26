"""solver 测试集的公共夹具。

整个子集需要 z3；未安装时整体跳过。``random_lia_instance`` 生成有界的
LIA 单目标实例，用于 oracle 一致性测试（与 z3-native ``Optimize`` 对比）。
"""

from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")  # 整个 solver 测试集需要 z3


def random_lia_instance(seed: int, n_vars: int = 4, n_constraints: int = 6,
                        sense_min: bool = True):
    """构造一个有界的 LIA 单目标实例。

    返回 ``(hard, variables, objective, sense_min)``：
    - ``hard``：``list[z3.BoolRef]`` 硬约束（含每变量 ``0 <= x <= 20`` 盒约束，保证有界）；
    - ``variables``：``list[z3.ArithRef]``；
    - ``objective``：``z3.ArithRef`` 线性目标；
    - ``sense_min``：是否最小化。
    """
    import random

    rng = random.Random(seed)
    xs = [z3.Int(f"x{i}") for i in range(n_vars)]
    hard = []
    for x in xs:
        hard.append(x >= 0)
        hard.append(x <= 20)
    for _ in range(n_constraints):
        coeffs = [rng.randint(-3, 3) for _ in xs]
        rhs = rng.randint(0, 40)
        expr = z3.Sum([c * x for c, x in zip(coeffs, xs)])
        hard.append(expr <= rhs)
    obj = z3.Sum([rng.randint(1, 4) * x for x in xs])
    return hard, xs, obj, sense_min


@pytest.fixture
def lia_instance():
    return random_lia_instance(seed=0)
