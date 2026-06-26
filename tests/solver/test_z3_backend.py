from __future__ import annotations

from fractions import Fraction

import pytest

z3 = pytest.importorskip("z3")

from omt_branching.solver.interfaces import Sense
from omt_branching.solver.z3_backend import Z3Backend


def test_solve_sat_and_unsat():
    b = Z3Backend()
    x = z3.Int("x")
    assert b.solve(z3.And(x > 0, x < 3)) is not None
    assert b.solve(z3.And(x > 3, x < 1)) is None


def test_optimize_matches_native_min_max():
    b = Z3Backend()
    x, y = z3.Int("x"), z3.Int("y")
    hard = z3.And(x + y <= 10, x >= 0, y >= 0)
    m, v = b.optimize(hard, 2 * x + y, Sense.MAX)
    assert v == 20
    m2, v2 = b.optimize(hard, x + y, Sense.MIN)
    assert v2 == 0


def test_value_int_and_rational():
    b = Z3Backend()
    x = z3.Real("x")
    m = b.solve(2 * x == 1)
    assert b.value(m, x) == Fraction(1, 2)


def test_better_constraint_min():
    b = Z3Backend()
    x = z3.Int("x")
    # better-than-5 for MIN means x < 5; x==4 satisfies, x==6 does not
    c = b.better(x, 5, Sense.MIN)
    assert b.solve(z3.And(c, x == 4)) is not None
    assert b.solve(z3.And(c, x == 6)) is None


def test_better_constraint_max():
    b = Z3Backend()
    x = z3.Int("x")
    c = b.better(x, 5, Sense.MAX)
    assert b.solve(z3.And(c, x == 6)) is not None
    assert b.solve(z3.And(c, x == 4)) is None


def test_le_ge_and_conjoin_negate_top():
    b = Z3Backend()
    x = z3.Int("x")
    assert b.solve(z3.And(b.le(x, 3), b.ge(x, 3), x == 3)) is not None
    assert b.solve(b.conjoin(b.le(x, 1), b.ge(x, 5))) is None
    assert b.solve(b.conjoin()) is not None          # empty conjunction = True
    assert b.solve(z3.And(b.negate(x == 0), x == 0)) is None
    assert b.solve(b.top()) is not None
