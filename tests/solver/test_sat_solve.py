from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")

from omt_branching.solver.sat_instances import generate_php
from omt_branching.solver.sat_solve import solve_sat_with_decider
from omt_branching.solver.propagator_snapshot import atom_key


def test_vsids_arm_has_conflicts_and_correct():
    atoms, clauses = generate_php(6)                       # PHP(7,6) UNSAT
    r = solve_sat_with_decider(clauses, atoms, decider_factory=None)
    assert r["result"] == "unsat"                          # 正确性
    assert r["conflicts"] > 100                            # 附 propagator -> 纯 CDCL 大量冲突
    assert r["decisions"] == 0                             # VSIDS 臂我们不覆盖


def test_override_arm_controls_and_correct():
    atoms, clauses = generate_php(6)
    r = solve_sat_with_decider(
        clauses, atoms,
        decider_factory=lambda a: (lambda und, asg: (min(und), True)))
    assert r["result"] == "unsat"
    assert r["decisions"] > 0                               # 我们强制了决策
