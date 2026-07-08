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


def generate_hard_smt_lia(n_vars: int = 8, n_disj: int = 30, k: int = 3,
                          ub: int = 6, chi: int = 4, seed: int = 0):
    """紧随机 SMT(LIA)：``n_vars`` 整数变量 + 盒约束；``n_disj`` 个 ``k`` 元析取(线性原子)。
    系数 [-chi,chi]、小域 [0,ub] 使布尔搜索成瓶颈（附 propagator 数百 conflicts）。返回
    (atoms, clauses)，``atoms``=出现的理论原子（供 propagator 分支）。
    """
    from omt_branching.solver.propagator_snapshot import collect_atoms

    rng = random.Random(seed)
    xs = [z3.Int(f"y{i}") for i in range(n_vars)]
    clauses = []
    for x in xs:
        clauses.append(z3.Or(x >= 0))          # 盒下界（原子形式，便于 collect_atoms 收录）
        clauses.append(z3.Or(x <= ub))
    for _ in range(n_disj):
        lits = []
        for _ in range(k):
            c = [rng.randint(-chi, chi) for _ in range(n_vars)]
            if all(v == 0 for v in c):
                c[rng.randrange(n_vars)] = 1
            lhs = z3.Sum([cc * x for cc, x in zip(c, xs)])
            b = rng.randint(-ub, ub * chi)
            lits.append(lhs <= b if rng.random() < 0.5 else lhs >= b)
        clauses.append(z3.Or(*lits))
    atoms = collect_atoms(clauses)
    return atoms, clauses


__all__ = ["generate_php", "generate_rand_3sat", "generate_hard_smt_lia"]
