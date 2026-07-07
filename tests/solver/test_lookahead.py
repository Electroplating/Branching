from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")

from omt_branching.solver.lookahead import lookahead_scores, LookaheadConfig
from omt_branching.solver.propagator_snapshot import atom_key


def test_propagating_atom_outranks_isolated():
    x = [z3.Int(f"x{i}") for i in range(3)]
    a, b, c = x[0] >= 5, x[1] <= 2, x[2] >= 3
    hard = [x[0] >= 0, x[0] <= 10, x[1] >= 0, x[1] <= 10, x[2] >= 0, x[2] <= 10,
            z3.Or(a, b), z3.Or(z3.Not(a), c), z3.Or(b, c)]
    sc, ph = lookahead_scores(hard, atoms=[a, b, c])
    # a 两侧都传播(a=T->c, a=F->b)，b 相对孤立 -> score(a) > score(b)
    assert sc[atom_key(a)] > sc[atom_key(b)]
    assert atom_key(a) in ph and isinstance(ph[atom_key(a)], bool)


def test_failed_literal_gets_sentinel():
    x = z3.Int("x")
    # a: x>=8, 但另有 x<=3 硬约束 -> 假设 a=True 不可行 -> a 被强制为假(大哨兵)
    a = x >= 8
    hard = [x >= 0, x <= 10, x <= 3, z3.Or(a, x >= 1)]
    sc, ph = lookahead_scores(hard, atoms=[a], config=LookaheadConfig(sentinel=1e6))
    assert sc[atom_key(a)] >= 1e6      # failed literal
    assert ph[atom_key(a)] is False    # 可行侧是 a=False


def test_build_lookahead_examples_has_bool_labels():
    from omt_branching.solver import generate_bool_lia_dataset
    from omt_branching.solver.training_data import build_lookahead_examples
    from omt_branching.interfaces import NodeType

    ds = generate_bool_lia_dataset(6, seed=3, min_vars=5, max_vars=6)
    exs = build_lookahead_examples(ds)
    assert exs and any(e.bool_target_scores for e in exs)
    e = next(e for e in exs if e.bool_target_scores)
    n_bool = e.graph.num_nodes(NodeType.BOOL_VAR)
    assert all(0 <= k < n_bool for k in e.bool_target_scores)
    assert e.phase_targets   # phase 标签也在
