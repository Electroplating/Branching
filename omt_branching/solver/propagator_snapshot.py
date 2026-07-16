"""从 z3 布尔公式构造 SolverSnapshot（供 UserPropagator 学习分支）。

抽取原子（比较原子 + 布尔常量）与**子句共现图**（每个顶层 assertion 的原子构成一个 clause），
配合 propagator 提供的动态赋值/统计，喂给现有 GNN 策略。子句图是学习"好决策"的关键结构
（见 spec §5：无子句图极可能学不动）。

性能（阶段 1+2）：
1. ``atom_key``：对仍存活的 z3 AST 按 ``id(expr)`` 缓存 ``str(e)``（Python 对象 id，非
   ``get_id()``——后者在 AST 回收后会复用，不能做全局键）。
2. 固定 ``assertions`` 的静态结构 LRU 缓存；骨架钉住 assertion/原子引用以防 id 复用；
   refocus 时只灌 assignment / search_state。``_linear`` 仅在冷建静态骨架时用局部缓存。
"""
from __future__ import annotations

import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import z3

from omt_branching.input.solver_state import (
    BooleanVarInfo, ClauseInfo, NumericVarInfo, SearchStateInfo, SolverSnapshot,
    TheoryAtomInfo,
)
from omt_branching.solver.extractor import _map_kind, _merge

_CMP = {z3.Z3_OP_LE, z3.Z3_OP_LT, z3.Z3_OP_GE, z3.Z3_OP_GT, z3.Z3_OP_EQ}

# Python 对象 id -> str。键为 id(expr)，值旁路钉住 expr 本身，避免 GC 后 id 复用脏读。
_EXPR_STR_CACHE: dict[int, tuple[object, str]] = {}
_EXPR_STR_CACHE_MAX = 100_000

# 静态 snapshot 骨架 LRU
_STATIC_CACHE: OrderedDict[tuple, "_StaticBoolSnapshot"] = OrderedDict()
_STATIC_CACHE_MAX = 64


def atom_key(e) -> str:
    """z3 原子的稳定字符串键。

    对外语义仍为 ``str(e)``；进程内按 ``id(e)`` 缓存并钉住 ``e``，使同一次求解中
    对同一 ExprRef 的重复调用不再反复字符串化。不用 ``get_id()``：Z3 会在 AST
    回收后复用该 id。
    """
    pid = id(e)
    hit = _EXPR_STR_CACHE.get(pid)
    if hit is not None and hit[0] is e:
        return hit[1]
    s = str(e)
    if len(_EXPR_STR_CACHE) >= _EXPR_STR_CACHE_MAX:
        for drop in list(_EXPR_STR_CACHE.keys())[: _EXPR_STR_CACHE_MAX // 2]:
            _EXPR_STR_CACHE.pop(drop, None)
    _EXPR_STR_CACHE[pid] = (e, s)
    return s


def clear_bool_snapshot_cache() -> None:
    """清空静态结构 / 字符串缓存（测试或长跑内存回收用）。"""
    _EXPR_STR_CACHE.clear()
    _STATIC_CACHE.clear()


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
    seen_ids: set[int] = set()  # 子句内用 get_id 去重（assertion 存活期间安全）

    def add(e):
        atom, pos = _lit(e)
        if _is_atom(atom):
            eid = atom.get_id()
            if eid not in seen_ids:
                seen_ids.add(eid)
                lits.append((atom_key(atom), pos))
        else:
            for ch in atom.children():
                add(ch)

    if z3.is_or(assertion):
        for ch in assertion.children():
            add(ch)
    else:
        add(assertion)
    return lits


def _linear(e, cache: dict | None = None) -> tuple[dict, float]:
    """把线性算术表达式分解为 ``(变量名->系数, 常数)``；变量键用 ``atom_key``。

    ``cache`` 须为**单次建图局部**字典（键 ``get_id``）。Z3 会在 AST 回收后复用
    ``get_id``，故禁止进程级长期缓存。单次 ``_build_static`` 内相关子式均存活，安全。
    未传入 ``cache`` 时使用临时空字典（便于单测；无跨调用复用）。
    """
    if cache is None:
        cache = {}
    eid = e.get_id()
    cached = cache.get(eid)
    if cached is not None:
        return cached

    if z3.is_int_value(e):
        result = ({}, float(e.as_long()))
        cache[eid] = result
        return result
    if z3.is_rational_value(e):
        # 整数真除避免大分子/分母场景下的浮点溢出。
        result = ({}, e.numerator_as_long() / e.denominator_as_long())
        cache[eid] = result
        return result
    if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED and z3.is_arith(e):
        result = ({atom_key(e): 1.0}, 0.0)
        cache[eid] = result
        return result
    op = e.decl().kind()
    children = e.children()
    if op == z3.Z3_OP_ADD:
        coeffs: dict = {}
        const = 0.0
        for ch in children:
            cc, ck = _linear(ch, cache)
            _merge(coeffs, cc)
            const += ck
        result = (coeffs, const)
        cache[eid] = result
        return result
    if op == z3.Z3_OP_SUB:
        coeffs, const = _linear(children[0], cache)
        coeffs = dict(coeffs)
        for ch in children[1:]:
            cc, ck = _linear(ch, cache)
            _merge(coeffs, cc, -1.0)
            const -= ck
        result = (coeffs, const)
        cache[eid] = result
        return result
    if op == z3.Z3_OP_UMINUS:
        cc, ck = _linear(children[0], cache)
        result = ({v: -a for v, a in cc.items()}, -ck)
        cache[eid] = result
        return result
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
            result = ({}, scale)
            cache[eid] = result
            return result
        if len(rest) == 1:
            cc, ck = _linear(rest[0], cache)
            result = ({v: a * scale for v, a in cc.items()}, ck * scale)
            cache[eid] = result
            return result
        warnings.warn(f"_linear 遇到无法分解的表达式，退化为空系数: {e}")
        result = ({}, 0.0)  # 两个非常量因子相乘：非线性，本任务原子不会触发
        cache[eid] = result
        return result
    warnings.warn(f"_linear 遇到无法分解的表达式，退化为空系数: {e}")
    result = ({}, 0.0)  # 未识别的算术形状：退化为 0，不阻断建图
    cache[eid] = result
    return result


@dataclass
class _StaticBoolSnapshot:
    """与 assignment / search_state 无关的 snapshot 骨架（可跨 refocus 复用）。"""

    # 钉住顶层 assertion，保证 cache key 中的 get_id 不与后续临时式冲突
    pinned_assertions: tuple = field(repr=False)
    amap: dict
    atom_keys: list[str]
    clauses: list
    theory_atoms: list
    numeric_vars: list
    occ: dict
    pos: dict
    neg: dict


def _assertions_cache_key(assertions) -> tuple:
    if not assertions:
        return (0,)
    ctx = id(assertions[0].ctx) if hasattr(assertions[0], "ctx") else 0
    return (ctx, *(a.get_id() for a in assertions))


def _build_static(assertions) -> _StaticBoolSnapshot:
    """从 assertions 抽取静态结构（atoms / clauses / theory / 度统计）。"""
    atoms = collect_atoms(assertions)
    amap = {atom_key(t): t for t in atoms}
    atom_keys = list(amap.keys())

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

    # 理论原子结构特征（LIA）：numeric_var -> theory_atom -> bool_var。
    # 纯 SAT 布尔常量被守卫跳过，theory_atoms / numeric_vars 保持为空。
    # _linear 局部缓存：仅本函数内有效，避免 get_id 复用脏读。
    linear_cache: dict = {}
    theory_atoms: list[TheoryAtomInfo] = []
    seen_vars: dict = {}
    for k, atom in amap.items():
        op = atom.decl().kind()
        if op not in _CMP or atom.num_args() < 2 or not z3.is_arith(atom.arg(0)):
            continue
        lhs_c, lhs_k = _linear(atom.arg(0), linear_cache)
        rhs_c, rhs_k = _linear(atom.arg(1), linear_cache)
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

    return _StaticBoolSnapshot(
        pinned_assertions=tuple(assertions),
        amap=amap,
        atom_keys=atom_keys,
        clauses=clauses,
        theory_atoms=theory_atoms,
        numeric_vars=numeric_vars,
        occ=occ,
        pos=pos,
        neg=neg,
    )


def _get_static(assertions) -> _StaticBoolSnapshot:
    key = _assertions_cache_key(assertions)
    hit = _STATIC_CACHE.get(key)
    if hit is not None:
        # 二次校验：get_id 复用时 pinned 引用会对不上
        if (
            len(hit.pinned_assertions) == len(assertions)
            and all(a is b for a, b in zip(hit.pinned_assertions, assertions))
        ):
            _STATIC_CACHE.move_to_end(key)
            return hit
        _STATIC_CACHE.pop(key, None)
    static = _build_static(assertions)
    _STATIC_CACHE[key] = static
    while len(_STATIC_CACHE) > _STATIC_CACHE_MAX:
        _STATIC_CACHE.popitem(last=False)
    return static


def build_bool_snapshot(assertions, assignment: Optional[dict] = None,
                        stats: Optional[dict] = None, snapshot_id: str = "prop"):
    """构造 SolverSnapshot。

    静态部分（原子/子句/理论特征）按 ``assertions`` 缓存；每次调用只根据
    ``assignment`` / ``stats`` 填充动态字段。
    """
    assignment = assignment or {}
    stats = stats or {}
    static = _get_static(list(assertions))

    bool_vars = [
        BooleanVarInfo(
            var_id=k,
            assignment=assignment.get(k),
            is_candidate=True,
            occurrence_count=static.occ[k],
            pos_count=static.pos[k],
            neg_count=static.neg[k],
        )
        for k in static.atom_keys
    ]
    search_state = SearchStateInfo(
        decision_level=int(stats.get("decisions", 0)),
        conflict_count=int(stats.get("conflicts", 0)),
        trail_length=len(assignment),
    )
    snap = SolverSnapshot(
        bool_vars=bool_vars,
        clauses=static.clauses,
        theory_atoms=static.theory_atoms,
        numeric_vars=static.numeric_vars,
        search_state=search_state,
        candidate_bool_ids=list(static.atom_keys),
        candidate_numeric_ids=[],
        snapshot_id=snapshot_id,
    )
    return snap, static.amap


__all__ = [
    "atom_key",
    "collect_atoms",
    "build_bool_snapshot",
    "clear_bool_snapshot_cache",
]
