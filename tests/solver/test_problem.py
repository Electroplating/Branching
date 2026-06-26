from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")

from omt_branching.solver.interfaces import Sense
from omt_branching.solver.z3_backend import Z3Backend
from omt_branching.solver.problem import GOMTProblem, Infeasible


def test_initial_state_has_incumbent_and_better_delta():
    b = Z3Backend()
    x = z3.Int("x")
    prob = GOMTProblem(hard_list=(x >= 0, x <= 5), objective=x, sense=Sense.MIN)
    st = prob.initial_state(b)
    assert st.incumbent is not None
    v = b.value(st.incumbent, x)
    assert 0 <= v <= 5
    # delta excludes the incumbent's value (must be strictly better)
    assert b.solve(z3.And(st.hard, st.delta, x == v)) is None
    assert st.tau == [st.delta]
    assert st.stats["sats"] == 0 and st.stats["splits"] == 0


def test_infeasible_raises():
    b = Z3Backend()
    x = z3.Int("x")
    prob = GOMTProblem(hard_list=(x > 0, x < 0), objective=x, sense=Sense.MIN)
    with pytest.raises(Infeasible):
        prob.initial_state(b)
