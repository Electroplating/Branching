"""困难 SAT 实例（供学习分支研究）：pigeonhole(UNSAT) + 相变点随机 3-SAT。

附 propagator 时 z3 关预处理走纯 CDCL，这些实例给出大量、受控的 conflicts（headroom）。
"""
from __future__ import annotations

import random

import z3


def generate_php(m: int):
    """PHP(m+1, m)：m+1 只鸽、m 个洞，**UNSAT**。返回 (atoms, clauses)。"""
    p = [[z3.Bool(f"p_{i}_{j}") for j in range(m)] for i in range(m + 1)]
    clauses = [z3.Or(*p[i]) for i in range(m + 1)]                 # 每鸽入某洞
    for j in range(m):
        for i1 in range(m + 1):
            for i2 in range(i1 + 1, m + 1):
                clauses.append(z3.Or(z3.Not(p[i1][j]), z3.Not(p[i2][j])))  # 无两鸽同洞
    atoms = [b for row in p for b in row]
    return atoms, clauses


def generate_rand_3sat(n: int, ratio: float = 4.26, seed: int = 0):
    """相变点附近随机 3-SAT。返回 (atoms, clauses)。"""
    rng = random.Random(seed)
    xs = [z3.Bool(f"v{i}") for i in range(n)]
    clauses = []
    for _ in range(int(n * ratio)):
        idx = rng.sample(range(n), 3)
        clauses.append(z3.Or([xs[i] if rng.random() < 0.5 else z3.Not(xs[i]) for i in idx]))
    return xs, clauses


__all__ = ["generate_php", "generate_rand_3sat"]
