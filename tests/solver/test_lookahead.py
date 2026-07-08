from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")
torch = pytest.importorskip("torch")

from omt_branching.model.policy import BranchingPolicy
from omt_branching.solver.lookahead import (
    LookaheadConfig, lookahead_scores, build_lookahead_example,
    build_lookahead_examples, decide_bool_hit,
)
from omt_branching.solver.instance_gen import generate_bool_lia_dataset
from omt_branching.solver.propagator_snapshot import atom_key


def test_lookahead_scores_forced_literal_gets_high_score():
    # a := (x >= 5)；断言强制 a 为真（x >= 6 => x >= 5），故 ¬a 一侧应 UNSAT。
    x = z3.Int("x")
    a, b = x >= 5, x <= 20
    asserts = [x >= 6, x <= 100, z3.Or(a, b)]
    scores, phases = lookahead_scores(asserts)
    ka = atom_key(a)
    assert ka in scores
    # a 被蕴含（¬a 一侧 UNSAT）=> 前瞻记大分，相位为真（探被强制的方向）。
    assert scores[ka] >= LookaheadConfig().sentinel - 1
    assert phases[ka] is True


def test_lookahead_scores_symmetric_atom_finite():
    x, y = z3.Int("x"), z3.Int("y")
    a = x >= y
    asserts = [x >= 0, x <= 10, y >= 0, y <= 10, z3.Or(a, x <= 3)]
    scores, phases = lookahead_scores(asserts)
    ka = atom_key(a)
    # 两侧都可行的原子拿到有限分（非 sentinel）。
    assert ka in scores
    assert 0.0 <= scores[ka] < LookaheadConfig().sentinel


def test_build_lookahead_example_maps_local_indices():
    inst = generate_bool_lia_dataset(1, seed=3, min_vars=4, max_vars=4)[0]
    hard = list(inst.hard)
    ex = build_lookahead_example(hard)
    assert ex is not None
    # 标签键必须是图内局部索引，且落在 bool 分数张量范围内。
    from omt_branching.interfaces import NodeType
    bmap = ex.graph.id_maps.get(NodeType.BOOL_VAR, {})
    assert bmap
    assert all(0 <= k < len(bmap) for k in ex.bool_target_scores)
    assert ex.bool_target_scores  # 非空


def test_decide_bool_hit_returns_bool_or_none():
    inst = generate_bool_lia_dataset(1, seed=7, min_vars=4, max_vars=4)[0]
    hit = decide_bool_hit(BranchingPolicy(), list(inst.hard))
    assert hit is None or isinstance(hit, bool)


def test_build_lookahead_examples_batch():
    insts = generate_bool_lia_dataset(3, seed=1, min_vars=4, max_vars=4)
    examples = build_lookahead_examples([list(i.hard) for i in insts])
    assert 1 <= len(examples) <= 3
