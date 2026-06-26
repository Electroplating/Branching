from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")

from omt_branching.solver.interfaces import Sense
from omt_branching.solver.bridge import NeuralGOMTSolver, BridgeConfig, solve_native
from tests.solver.conftest import random_lia_instance


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("sense", [Sense.MIN, Sense.MAX])
def test_neural_gomt_matches_native(seed, sense):
    hard, xs, obj, _ = random_lia_instance(seed=seed)
    native = solve_native(tuple(hard), obj, sense)
    res = NeuralGOMTSolver().solve(tuple(hard), obj, sense)
    assert res.optimal is True
    assert res.value == native


def test_hybrid_mode_also_matches():
    hard, xs, obj, _ = random_lia_instance(seed=3)
    native = solve_native(tuple(hard), obj, Sense.MAX)
    res = NeuralGOMTSolver(config=BridgeConfig(f_sat_mode="hybrid")).solve(
        tuple(hard), obj, Sense.MAX)
    assert res.value == native


def test_baseline_strategy_matches():
    hard, xs, obj, _ = random_lia_instance(seed=5)
    native = solve_native(tuple(hard), obj, Sense.MIN)
    res = NeuralGOMTSolver(config=BridgeConfig(strategy="baseline")).solve(
        tuple(hard), obj, Sense.MIN)
    assert res.optimal and res.value == native
