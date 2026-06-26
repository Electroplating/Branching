from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")

from omt_branching.solver.interfaces import Sense
from omt_branching.solver.z3_backend import Z3Backend
from omt_branching.solver.problem import GOMTProblem
from omt_branching.solver.calculus import GOMTSolver
from omt_branching.solver.strategy import BaselineStrategy, NeuralStrategy
from omt_branching.service import BranchingPolicyService
from tests.solver.conftest import random_lia_instance


def _native_opt(hard, obj, sense_min):
    o = z3.Optimize()
    for c in hard:
        o.add(c)
    h = o.minimize(obj) if sense_min else o.maximize(obj)
    assert o.check() == z3.sat
    return (o.lower(h) if sense_min else o.upper(h)).as_long()


@pytest.mark.parametrize("seed", range(5))
def test_baseline_reaches_optimum(seed):
    b = Z3Backend()
    hard, xs, obj, _ = random_lia_instance(seed=seed)
    prob = GOMTProblem(hard_list=tuple(hard), objective=obj, sense=Sense.MIN)
    res = GOMTSolver(prob, b, BaselineStrategy(prob)).run()
    assert res.optimal and res.value == _native_opt(hard, obj, True)


@pytest.mark.parametrize("seed", range(5))
def test_neural_untrained_reaches_optimum(seed):
    b = Z3Backend()
    hard, xs, obj, _ = random_lia_instance(seed=seed)
    prob = GOMTProblem(hard_list=tuple(hard), objective=obj, sense=Sense.MAX)
    service = BranchingPolicyService()                # untrained policy
    res = GOMTSolver(prob, b, NeuralStrategy(prob, service)).run()
    # soundness: optimum independent of policy quality (Thm 1)
    assert res.optimal and res.value == _native_opt(hard, obj, False)


def test_neural_actually_splits():
    b = Z3Backend()
    hard, xs, obj, _ = random_lia_instance(seed=2)
    prob = GOMTProblem(hard_list=tuple(hard), objective=obj, sense=Sense.MAX)
    res = GOMTSolver(prob, b, NeuralStrategy(prob, BranchingPolicyService())).run()
    assert res.stats["splits"] >= 1
