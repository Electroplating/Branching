"""SAT look-ahead 教师：假设某原子，用 z3 consequences 计"强制了多少其他原子"（传播强度），
作 imitation 监督标签。传播强度是**子句共现图的函数**——正是 GNN 已见的特征，故可学
（对比 LIA 分离度需缺失的 LP 特征）。
"""
from __future__ import annotations

from dataclasses import dataclass

import z3

from omt_branching.solver.propagator_snapshot import atom_key, collect_atoms


@dataclass(frozen=True)
class LookaheadConfig:
    max_atoms: int = 32
    sentinel: float = 1e6
    eps: float = 1e-9


def _strip_not(e):
    while z3.is_not(e):
        e = e.arg(0)
    return e


def _count_other(imps, self_key: str) -> int:
    """统计蕴含到的**其他**原子数（剥离 Not，排除自身/双重否定）。"""
    seen = set()
    for imp in imps:
        cons = imp.arg(1) if (z3.is_implies(imp) and imp.num_args() == 2) else imp
        k = atom_key(_strip_not(cons))
        if k != self_key:
            seen.add(k)
    return len(seen)


def lookahead_scores(assertions, atoms=None, config: LookaheadConfig = LookaheadConfig()):
    atom_exprs = list(atoms) if atoms is not None else collect_atoms(list(assertions))
    atom_exprs = atom_exprs[: config.max_atoms]
    s = z3.Solver()
    s.add(*assertions)

    scores: dict = {}
    phases: dict = {}
    for a in atom_exprs:
        k = atom_key(a)
        try:
            res_t, imp_t = s.consequences([a], atom_exprs)
            res_f, imp_f = s.consequences([z3.Not(a)], atom_exprs)
        except z3.Z3Exception:
            continue
        t_unsat = res_t == z3.unsat
        f_unsat = res_f == z3.unsat
        if t_unsat and f_unsat:
            continue                      # 两侧皆不可行：矛盾/无关，跳过
        if t_unsat:                       # a=True 不可行 -> a 被强制为假
            scores[k] = config.sentinel
            phases[k] = False
            continue
        if f_unsat:
            scores[k] = config.sentinel
            phases[k] = True
            continue
        pt = _count_other(imp_t, k)
        pf = _count_other(imp_f, k)
        scores[k] = (pt + 1.0) * (pf + 1.0)   # march 风格 product：两侧都传播多者优
        phases[k] = pt >= pf                  # 先探传播更多的一侧
    return scores, phases


__all__ = ["LookaheadConfig", "lookahead_scores"]
