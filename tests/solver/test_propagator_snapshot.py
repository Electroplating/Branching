from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")

from omt_branching.solver.propagator_snapshot import (
    atom_key, collect_atoms, build_bool_snapshot,
)


def test_collect_atoms_and_clause_cooccurrence():
    x = z3.Int("x")
    a, b, c = x >= 5, x <= 2, z3.Bool("c")
    asserts = [z3.Or(a, b), z3.Or(c, z3.Not(a))]
    atoms = collect_atoms(asserts)
    keys = {atom_key(t) for t in atoms}
    assert {atom_key(a), atom_key(b), atom_key(c)} <= keys

    snap, amap = build_bool_snapshot(asserts)
    bkeys = {bv.var_id for bv in snap.bool_vars}
    assert {atom_key(a), atom_key(b), atom_key(c)} <= bkeys
    # 每个顶层 assertion 的原子共现为一个 clause
    assert len(snap.clauses) == 2
    lits0 = {vid for vid, _ in snap.clauses[0].literals}
    assert lits0 == {atom_key(a), atom_key(b)}
    # ¬a 的极性为 False
    pol = dict((vid, pos) for vid, pos in snap.clauses[1].literals)
    assert pol[atom_key(a)] is False
    # 映射能取回 z3 原子
    assert amap[atom_key(a)] is not None


def test_assignment_and_candidates():
    x = z3.Int("x")
    a, b = x >= 5, x <= 2
    snap, _ = build_bool_snapshot([z3.Or(a, b)], assignment={atom_key(a): True})
    amap = {bv.var_id: bv for bv in snap.bool_vars}
    assert amap[atom_key(a)].assignment is True
    assert amap[atom_key(b)].assignment is None
    assert set(snap.candidate_bool_ids) == {atom_key(a), atom_key(b)}


def test_build_bool_snapshot_theory_features():
    """SMT(LIA) 应产生理论原子/数值变量结构特征；纯 SAT 实例保持为空（不变量）。"""
    from omt_branching.solver.sat_instances import generate_hard_smt_lia, generate_rand_3sat
    from omt_branching.input.graph_builder import GraphBuilder
    from omt_branching.interfaces import EdgeType

    # SMT(LIA)：理论原子 + 数值变量被填充，且 var_coeffs 键 ∈ numeric_vars
    atoms, clauses = generate_hard_smt_lia(6, 12, 3, 6, 4, seed=3)
    snap, _ = build_bool_snapshot(clauses)
    assert snap.theory_atoms, "SMT(LIA) 应产生理论原子节点"
    assert snap.numeric_vars, "SMT(LIA) 应产生数值变量节点"
    numvar_ids = {n.num_var_id for n in snap.numeric_vars}
    ta = snap.theory_atoms[0]
    assert ta.var_coeffs and all(v in numvar_ids for v in ta.var_coeffs)  # variable_in_atom 边连得上
    assert ta.bool_var_id in {b.var_id for b in snap.bool_vars}           # atom_abstracted_by 边连得上

    # 端到端连边断言：结构特征真的进了图（atom_abstracted_by / variable_in_atom 边数 > 0）
    g = GraphBuilder().build(snap)
    assert g.num_edges(EdgeType.ATOM_ABSTRACTED_BY) > 0
    assert g.num_edges(EdgeType.VARIABLE_IN_ATOM) > 0

    # SAT 不变性：纯布尔常量实例不产生理论/数值节点（保护 SAT 正结果）
    sat_atoms, sat_clauses = generate_rand_3sat(20, seed=7)
    ssnap, _ = build_bool_snapshot(sat_clauses)
    assert ssnap.theory_atoms == [] and ssnap.numeric_vars == []


def test_linear_decomposition_branches():
    """直接单测 _linear 的 SUB/UMINUS/有理常数分支，覆盖 ADD-of-MUL 之外的形状。"""
    import z3
    from omt_branching.solver.propagator_snapshot import _linear
    x, y = z3.Ints("x y")
    # SUB: 3*x - 2*y  -> {x:3, y:-2}, const 0
    c, k = _linear(3 * x - 2 * y)
    assert c[str(x)] == 3.0 and c[str(y)] == -2.0 and k == 0.0
    # UMINUS: -x -> {x:-1}
    c, k = _linear(-x)
    assert c[str(x)] == -1.0
    # 有理常数 (Real): x/2 形式的系数与常数项
    r = z3.Real("r")
    c, k = _linear(r + z3.RealVal("3/2"))
    assert c[str(r)] == 1.0 and abs(k - 1.5) < 1e-9
