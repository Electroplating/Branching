"""从 z3 布尔公式构造 SolverSnapshot（供 UserPropagator 学习分支）。

抽取原子（比较原子 + 布尔常量）与**子句共现图**（每个顶层 assertion 的原子构成一个 clause），
配合 propagator 提供的动态赋值/统计，喂给现有 GNN 策略。子句图是学习"好决策"的关键结构
（见 spec §5：无子句图极可能学不动）。
"""
from __future__ import annotations

from typing import Optional

import z3

from omt_branching.input.solver_state import (
    BooleanVarInfo, ClauseInfo, NumericVarInfo, SearchStateInfo, SolverSnapshot,
    TheoryAtomInfo,
)
from omt_branching.solver.extractor import _map_kind, _merge

_CMP = {z3.Z3_OP_LE, z3.Z3_OP_LT, z3.Z3_OP_GE, z3.Z3_OP_GT, z3.Z3_OP_EQ}


def atom_key(e) -> str:
    """z3 原子的稳定字符串键（同一进程内对同一原子稳定）。"""
    return str(e)


def _is_atom(e) -> bool:
    if not z3.is_bool(e):
        return False
    op = e.decl().kind()
    if op in _CMP and e.num_args() >= 1 and z3.is_arith(e.arg(0)):
        return True
    return z3.is_const(e) and op == z3.Z3_OP_UNINTERPRETED


def _lit(e):
    """返回 (atom_expr, is_positive)；``Not(a)`` -> (a, False)。"""
    if z3.is_not(e):
        return e.arg(0), False
    return e, True


def _walk_atoms(e, out, seen):
    eid = e.get_id()
    if eid in seen:
        return
    seen.add(eid)
    if _is_atom(e):
        out.append(e)
        return
    if not z3.is_bool(e):
        return
    for ch in e.children():
        _walk_atoms(ch, out, seen)


def collect_atoms(assertions) -> list:
    out: list = []
    seen: set = set()
    dedup: dict = {}
    for a in assertions:
        _walk_atoms(a, out, seen)
    # 按 atom_key 去重、保序
    uniq = []
    for t in out:
        k = atom_key(t)
        if k not in dedup:
            dedup[k] = t
            uniq.append(t)
    return uniq


def _clause_literals(assertion):
    """把一个顶层 assertion 拉平成 (atom_key, is_positive) 列表（Or 展开，其余取自身原子）。"""
    lits = []
    seen = set()

    def add(e):
        atom, pos = _lit(e)
        if _is_atom(atom):
            k = atom_key(atom)
            if k not in seen:
                seen.add(k)
                lits.append((k, pos))
        else:
            for ch in atom.children():
                add(ch)

    if z3.is_or(assertion):
        for ch in assertion.children():
            add(ch)
    else:
        add(assertion)
    return lits


def _linear(e) -> tuple[dict, float]:
    """把线性算术表达式分解为 ``(变量名->系数, 常数)``；变量键用 ``str(e)``。

    自包含版本（不依赖外部 ``var_exprs`` 登记表），供本模块抽取理论原子的
    ``var_coeffs``/``rhs`` 结构特征使用；算法与 ``extractor.Z3SnapshotExtractor._linear``
    一致（ADD/SUB/UMINUS/MUL(常数×子式)/整数与有理常数），但省略求解器句柄登记——
    这里只需要给 GNN 的图特征。``generate_hard_smt_lia`` 构造的原子保证线性，
    未识别的形状（理论上不会出现）退化为空系数+0 常数，不阻断建图。
    """
    if z3.is_int_value(e):
        return {}, float(e.as_long())
    if z3.is_rational_value(e):
        # 整数真除避免大分子/分母场景下的浮点溢出。
        return {}, e.numerator_as_long() / e.denominator_as_long()
    if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED and z3.is_arith(e):
        return {str(e): 1.0}, 0.0
    op = e.decl().kind()
    children = e.children()
    if op == z3.Z3_OP_ADD:
        coeffs: dict = {}
        const = 0.0
        for ch in children:
            cc, ck = _linear(ch)
            _merge(coeffs, cc)
            const += ck
        return coeffs, const
    if op == z3.Z3_OP_SUB:
        coeffs, const = _linear(children[0])
        coeffs = dict(coeffs)
        for ch in children[1:]:
            cc, ck = _linear(ch)
            _merge(coeffs, cc, -1.0)
            const -= ck
        return coeffs, const
    if op == z3.Z3_OP_UMINUS:
        cc, ck = _linear(children[0])
        return {v: -a for v, a in cc.items()}, -ck
    if op == z3.Z3_OP_MUL:
        scale = 1.0
        rest = []
        for ch in children:
            if z3.is_int_value(ch):
                scale *= float(ch.as_long())
            elif z3.is_rational_value(ch):
                scale *= float(ch.numerator_as_long()) / float(ch.denominator_as_long())
            else:
                rest.append(ch)
        if not rest:
            return {}, scale
        if len(rest) == 1:
            cc, ck = _linear(rest[0])
            return {v: a * scale for v, a in cc.items()}, ck * scale
        return {}, 0.0  # 两个非常量因子相乘：非线性，本任务原子不会触发
    return {}, 0.0  # 未识别的算术形状：退化为 0，不阻断建图


def build_bool_snapshot(assertions, assignment: Optional[dict] = None,
                        stats: Optional[dict] = None, snapshot_id: str = "prop"):
    assignment = assignment or {}
    stats = stats or {}
    atoms = collect_atoms(assertions)
    amap = {atom_key(t): t for t in atoms}

    # per-atom 子句统计（度/极性）——布尔节点本身无区分特征(赋值/默认全同)，若不给结构特征，
    # GNN 会把所有原子塌缩成相同 embedding、分数相同、梯度为零、无法学。子句度+极性是区分指纹。
    occ = {k: 0 for k in amap}
    pos = {k: 0 for k in amap}
    neg = {k: 0 for k in amap}
    clauses = []
    for i, a in enumerate(assertions):
        lits = [(k, p) for (k, p) in _clause_literals(a) if k in amap]
        for k, p in lits:
            occ[k] += 1
            if p:
                pos[k] += 1
            else:
                neg[k] += 1
        if lits:
            clauses.append(ClauseInfo(clause_id=f"c{i}", literals=lits))

    bool_vars = [
        BooleanVarInfo(var_id=k, assignment=assignment.get(k), is_candidate=True,
                       occurrence_count=occ[k], pos_count=pos[k], neg_count=neg[k])
        for k in amap
    ]

    # 理论原子结构特征（LIA 教训修复）：子句共现图丢掉了共享变量结构——两个原子是否共享
    # x0/x3 无法从"是否共现于同一子句"看出，导致 GNN 把所有候选塌缩成同一 embedding、
    # 排序梯度恒为 0。这里为每个比较原子(a^T x <=/>=/= b)追加 var_coeffs/rhs，并登记
    # 数值变量节点，使 numeric_var -> theory_atom -> bool_var 的消息传递能感知共享变量。
    # 布尔常量原子（SAT：z3.Bool 常量，无 arith 参数）被下面的守卫跳过，theory_atoms/
    # numeric_vars 保持为空 —— 这是保护 SAT 正结果的核心不变量。
    theory_atoms: list[TheoryAtomInfo] = []
    seen_vars: dict = {}    # 有序集合（dict 保序、O(1) 去重）
    for k, atom in amap.items():
        op = atom.decl().kind()
        if op not in _CMP or atom.num_args() < 2 or not z3.is_arith(atom.arg(0)):
            continue        # 非比较原子 / 非算术比较（含纯 SAT 布尔常量）：跳过
        lhs_c, lhs_k = _linear(atom.arg(0))
        rhs_c, rhs_k = _linear(atom.arg(1))
        coeffs: dict = {}
        _merge(coeffs, lhs_c)
        _merge(coeffs, rhs_c, -1.0)
        coeffs = {v: c for v, c in coeffs.items() if c != 0.0}
        rhs_val = rhs_k - lhs_k
        theory_atoms.append(TheoryAtomInfo(
            atom_id=k, bool_var_id=k, kind=_map_kind(op),
            var_coeffs=coeffs, rhs=float(rhs_val),
        ))
        for v in coeffs:
            seen_vars.setdefault(v, None)
    numeric_vars = [NumericVarInfo(num_var_id=v, is_integer=True) for v in seen_vars]

    search_state = SearchStateInfo(
        decision_level=int(stats.get("decisions", 0)),
        conflict_count=int(stats.get("conflicts", 0)),
        trail_length=len(assignment),
    )
    snap = SolverSnapshot(
        bool_vars=bool_vars, clauses=clauses, theory_atoms=theory_atoms,
        numeric_vars=numeric_vars,
        search_state=search_state,
        candidate_bool_ids=list(amap.keys()), candidate_numeric_ids=[],
        snapshot_id=snapshot_id,
    )
    return snap, amap


__all__ = ["atom_key", "collect_atoms", "build_bool_snapshot"]
