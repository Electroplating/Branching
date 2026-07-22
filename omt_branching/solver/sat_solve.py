"""单次 SAT 可满足性检查 harness：**恒附** LearnedDecidePropagator（关 z3 预处理→纯 CDCL），
两臂公平——decider_factory=None => defer-always（z3 自身 VSIDS）；给出 decider => 学习覆盖。
指标 conflicts 是分支质量的直接度量（无 OMT 回路稀释）。
"""
from __future__ import annotations

import z3

from omt_branching.solver.propagator import LearnedDecidePropagator


def _stat(s, key):
    st = s.statistics()
    for k in st.keys():
        if k == key:
            return st.get_key_value(k)
    return 0


def solve_sat_with_decider(assertions, atoms, decider_factory=None) -> dict:
    s = z3.Solver()
    if decider_factory is None:
        decider = lambda und, asg: None            # defer-always = VSIDS 臂
    else:
        decider = decider_factory(list(assertions))
    prop = LearnedDecidePropagator(s, atoms, decider)   # 恒附 -> 关预处理
    s.add(*assertions)
    res = s.check()
    return {
        "result": "sat" if res == z3.sat else ("unsat" if res == z3.unsat else "unknown"),
        "conflicts": _stat(s, "conflicts"),
        "decisions": prop.n_decisions,
        "on_decide": prop.n_on_decide,
        "next_split": prop.n_next_split,
        "defer": prop.n_defer,
        "rlimit": _stat(s, "rlimit count"),
    }


__all__ = ["solve_sat_with_decider"]
