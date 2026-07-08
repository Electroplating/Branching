from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")

from omt_branching.solver.sat_instances import generate_php, generate_rand_3sat


def test_php_is_unsat_and_shaped():
    atoms, clauses = generate_php(4)              # PHP(5,4)
    assert len(atoms) == 5 * 4                     # (m+1)*m 命题
    s = z3.Solver(); s.add(*clauses)
    assert s.check() == z3.unsat                   # 鸽笼原理 UNSAT
    assert all(z3.is_bool(a) for a in atoms)


def test_rand_3sat_reproducible():
    a1, c1 = generate_rand_3sat(30, 4.26, seed=1)
    a2, c2 = generate_rand_3sat(30, 4.26, seed=1)
    assert [str(a) for a in a1] == [str(a) for a in a2]
    assert len(c1) == int(30 * 4.26)
    assert all(z3.is_or(c) or z3.is_bool(c) for c in c1)


def test_hard_smt_lia_theory_atoms_and_conflicts():
    from omt_branching.solver.sat_instances import generate_hard_smt_lia
    from omt_branching.solver.sat_solve import solve_sat_with_decider
    atoms, clauses = generate_hard_smt_lia(n_vars=8, n_disj=30, k=3, ub=6, chi=4, seed=1)
    # 原子是线性算术比较（含加法/整数变量），非纯布尔常量
    assert len(atoms) > 20
    assert all(z3.is_bool(a) and a.num_args() >= 1 for a in atoms)   # 比较原子有参数
    # 附 propagator 关预处理 -> 大量 conflicts（headroom）
    r = solve_sat_with_decider(clauses, atoms, decider_factory=None)
    assert r["result"] in ("sat", "unsat")
    assert r["conflicts"] > 100                                       # 数百 conflicts
