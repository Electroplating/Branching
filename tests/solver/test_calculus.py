from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")

from omt_branching.solver.interfaces import Sense, SplitDecision
from omt_branching.solver.z3_backend import Z3Backend
from omt_branching.solver.problem import GOMTProblem
from omt_branching.solver.calculus import GOMTSolver, GOMTConfig
from tests.solver.conftest import random_lia_instance


class AlwaysResolve:
    """linear search 策略：从不 split，纯靠 F-Sat 收紧。"""

    def propose(self, state, backend):
        return SplitDecision.resolve()


def _native_opt(hard, obj, sense_min):
    o = z3.Optimize()
    for c in hard:
        o.add(c)
    h = o.minimize(obj) if sense_min else o.maximize(obj)
    assert o.check() == z3.sat
    ref = o.lower(h) if sense_min else o.upper(h)
    return ref.as_long()


def test_linear_search_reaches_native_optimum_min():
    b = Z3Backend()
    hard, xs, obj, _ = random_lia_instance(seed=1)
    prob = GOMTProblem(hard_list=tuple(hard), objective=obj, sense=Sense.MIN)
    res = GOMTSolver(prob, b, AlwaysResolve()).run()
    assert res.optimal is True
    assert res.value == _native_opt(hard, obj, sense_min=True)
    assert res.stats["steps"] >= 1


def test_split_path_with_objective_bisection():
    """手写一个 split 一次的策略，验证 F-Split 分支与 soundness。"""
    b = Z3Backend()
    x = z3.Int("x")
    prob = GOMTProblem(hard_list=(x >= 0, x <= 10), objective=x, sense=Sense.MAX)

    class SplitOnce:
        def __init__(self):
            self.done = False

        def propose(self, state, backend):
            if not self.done:
                self.done = True
                psi = state.top
                # ψ1 = ψ ∧ x>=6 (better half first), ψ2 = ψ ∧ x<=5
                return SplitDecision.split([
                    backend.conjoin(psi, backend.ge(x, 6)),
                    backend.conjoin(psi, backend.le(x, 5)),
                ])
            return SplitDecision.resolve()

    res = GOMTSolver(prob, b, SplitOnce(), GOMTConfig(check_invariants=True)).run()
    assert res.optimal is True
    assert res.value == 10
    assert res.stats["splits"] == 1
