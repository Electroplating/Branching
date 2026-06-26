"""简化的 OMT 求解器模拟器（solver replay）。

用于在无需真实 Z3/OptiMathSAT 的情况下，验证 GNN refocus 策略
对搜索过程的宏观影响。求解器维护：
- 布尔变量赋值 trail
- 整数变量当前界
- 已发生 conflict 计数
- 当前 incumbent / best bound

每次 decision 调用 BranchingPolicyService，比较 GNN refocus、VSIDS-only、
Oracle 三种策略的 decisions/conflicts/incumbent 演化。
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Hashable

from omt_branching.input.solver_state import (
    BooleanVarInfo,
    ClauseInfo,
    NumericVarInfo,
    ObjectiveInfo,
    SearchStateInfo,
    SolverSnapshot,
    TheoryAtomInfo,
)
from omt_branching.interfaces import AtomKind, ClauseKind, NodeType, SearchMode
from omt_branching.service import BranchingPolicyService


@dataclass
class SimProblem:
    """一个静态 OMT 问题实例。"""

    bool_vars: list[BooleanVarInfo]
    clauses: list[ClauseInfo]
    atoms: list[TheoryAtomInfo]
    numerics: list[NumericVarInfo]
    objective: ObjectiveInfo


@dataclass
class SimState:
    """求解器动态状态。"""

    assignment: dict[Hashable, bool] = field(default_factory=dict)
    num_lb: dict[Hashable, float] = field(default_factory=dict)
    num_ub: dict[Hashable, float] = field(default_factory=dict)
    decisions: int = 0
    conflicts: int = 0
    propagations: int = 0
    depth: int = 0
    incumbent: float = float("inf")


def _fresh_state(prob: SimProblem) -> SimState:
    return SimState(
        num_lb={n.num_var_id: n.lower_bound for n in prob.numerics if n.lower_bound is not None},
        num_ub={n.num_var_id: n.upper_bound for n in prob.numerics if n.upper_bound is not None},
    )


def _build_snapshot(prob: SimProblem, state: SimState, step: int) -> SolverSnapshot:
    """从当前模拟状态构造 SolverSnapshot。"""
    bool_vars = []
    for b in prob.bool_vars:
        ab = copy.copy(b)
        if b.var_id in state.assignment:
            ab.assignment = state.assignment[b.var_id]
            ab.is_candidate = False
        else:
            ab.assignment = None
            ab.is_candidate = True
        bool_vars.append(ab)

    numerics = []
    for n in prob.numerics:
        nv = copy.copy(n)
        nv.lower_bound = state.num_lb.get(n.num_var_id, n.lower_bound)
        nv.upper_bound = state.num_ub.get(n.num_var_id, n.upper_bound)
        # 模拟 LP 值取区间中点附近的随机点
        lo = nv.lower_bound or 0.0
        hi = nv.upper_bound or (lo + 10.0)
        mid = (lo + hi) / 2.0
        nv.lp_value = mid + random.uniform(-0.3, 0.3) * (hi - lo)
        nv.is_fractional = abs(nv.lp_value - round(nv.lp_value)) > 1e-3
        numerics.append(nv)

    objective = copy.copy(prob.objective)
    # 用当前界估算 incumbent
    est = 0.0
    for n in prob.numerics:
        coef = objective.var_coeffs.get(n.num_var_id, 0.0)
        val = state.num_lb.get(n.num_var_id, 0.0) if coef >= 0 else state.num_ub.get(n.num_var_id, 0.0)
        est += coef * val
    for vid, w in objective.soft_weights.items():
        if state.assignment.get(vid, False):
            est += 0.0
        else:
            est += w
    objective.incumbent = min(state.incumbent, est)
    objective.best_bound = est - random.uniform(0.5, 3.0)
    objective.gap = abs(objective.incumbent - objective.best_bound) / (abs(objective.incumbent) + 1e-6)

    state_info = SearchStateInfo(
        depth=state.depth,
        decision_level=state.depth,
        trail_length=len(state.assignment),
        conflict_count=state.conflicts,
        search_mode=SearchMode.LINEAR,
        time_budget_left=300.0,
    )

    return SolverSnapshot(
        bool_vars=bool_vars,
        clauses=prob.clauses,
        theory_atoms=prob.atoms,
        numeric_vars=numerics,
        objective=objective,
        search_state=state_info,
        snapshot_id=f"sim-{step}",
    )


def _apply_bool_decision(prob: SimProblem, state: SimState, var_id: Hashable, value: bool) -> bool:
    """执行布尔决策，模拟传播并检测冲突。返回是否冲突。"""
    state.assignment[var_id] = value
    state.decisions += 1
    state.depth += 1

    # 找到该变量对应的理论原子
    atom = next((a for a in prob.atoms if a.bool_var_id == var_id), None)
    if atom is None:
        return False

    # 根据原子真值施加数值变量界变化（简化版）
    for vid, coeff in atom.var_coeffs.items():
        if vid not in state.num_lb:
            continue
        if atom.kind == AtomKind.LE and value:
            # a^T x <= rhs => 对每个正系数变量收紧上界
            if coeff > 0:
                new_ub = atom.rhs / coeff
                state.num_ub[vid] = min(state.num_ub.get(vid, float("inf")), new_ub)
            elif coeff < 0:
                new_lb = atom.rhs / coeff
                state.num_lb[vid] = max(state.num_lb.get(vid, float("-inf")), new_lb)
        elif atom.kind == AtomKind.GE and value:
            if coeff > 0:
                new_lb = atom.rhs / coeff
                state.num_lb[vid] = max(state.num_lb.get(vid, float("-inf")), new_lb)
            elif coeff < 0:
                new_ub = atom.rhs / coeff
                state.num_ub[vid] = min(state.num_ub.get(vid, float("inf")), new_ub)
        state.propagations += 1

    # 检测冲突：任何变量下界 > 上界
    for vid in state.num_lb:
        if state.num_lb[vid] > state.num_ub.get(vid, float("inf")) + 1e-6:
            state.conflicts += 1
            state.depth = max(0, state.depth - 1)
            return True

    # 检测子句冲突（简化：所有文字为假）
    for c in prob.clauses:
        all_false = True
        satisfied = False
        for vid, pos in c.literals:
            val = state.assignment.get(vid)
            if val is None:
                all_false = False
                break
            if (pos and val) or (not pos and not val):
                satisfied = True
                break
        if all_false and not satisfied:
            state.conflicts += 1
            state.depth = max(0, state.depth - 1)
            return True

    return False


def _vsids_select(prob: SimProblem, state: SimState) -> tuple[Hashable, bool] | None:
    """原生 VSIDS 选择：候选中 activity 最高者。"""
    cands = [b for b in prob.bool_vars if b.var_id not in state.assignment]
    if not cands:
        return None
    best = max(cands, key=lambda b: b.vsids_activity)
    phase = best.phase_saved if best.phase_saved is not None else True
    return best.var_id, phase


def _gnn_select(
    prob: SimProblem, state: SimState, service: BranchingPolicyService, step: int
) -> tuple[Hashable, bool] | None:
    snap = _build_snapshot(prob, state, step)
    advice = service.advise(snap)
    if not advice.ranked_candidates:
        return _vsids_select(prob, state)
    top = advice.ranked_candidates[0]
    phase = advice.phase_suggestions.get(top, True)
    return top, phase


def _oracle_select(prob: SimProblem, state: SimState) -> tuple[Hashable, bool] | None:
    """使用 OracleBrancher 的 score 选择。"""
    from experiments.oracle import OracleBrancher

    snap = _build_snapshot(prob, state, 0)
    scores = OracleBrancher().bool_branch_scores(snap)
    cands = {k: v for k, v in scores.items() if k not in state.assignment}
    if not cands:
        return None
    top = max(cands, key=cands.get)
    phase = OracleBrancher().phase_labels(snap).get(top, True)
    return top, phase


@dataclass
class SimResult:
    strategy: str
    decisions: int
    conflicts: int
    propagations: int
    final_incumbent: float
    steps: int


def simulate(
    prob: SimProblem,
    strategy: str = "gnn",
    service: BranchingPolicyService | None = None,
    max_steps: int = 100,
    seed: int | None = None,
) -> SimResult:
    random.seed(seed)
    state = _fresh_state(prob)

    selector: Callable[[], tuple[Hashable, bool] | None]
    if strategy == "gnn":
        if service is None:
            raise ValueError("GNN strategy requires a BranchingPolicyService")
        selector = lambda: _gnn_select(prob, state, service, state.decisions)
    elif strategy == "oracle":
        selector = lambda: _oracle_select(prob, state)
    else:
        selector = lambda: _vsids_select(prob, state)

    for _ in range(max_steps):
        choice = selector()
        if choice is None:
            break
        var_id, phase = choice
        conflict = _apply_bool_decision(prob, state, var_id, phase)
        if conflict:
            # 简化回溯：清空最近赋值
            if state.assignment:
                state.assignment.popitem()
            state.depth = max(0, state.depth - 1)
        if state.decisions >= max_steps:
            break

    return SimResult(
        strategy=strategy,
        decisions=state.decisions,
        conflicts=state.conflicts,
        propagations=state.propagations,
        final_incumbent=state.incumbent,
        steps=state.decisions,
    )


def problem_from_snapshot(snap: SolverSnapshot) -> SimProblem:
    """把任意 SolverSnapshot 转成静态 SimProblem（忽略当前赋值）。"""
    bool_vars = []
    for b in snap.bool_vars:
        bc = copy.copy(b)
        bc.assignment = None
        bc.is_candidate = True
        bool_vars.append(bc)
    return SimProblem(
        bool_vars=bool_vars,
        clauses=[copy.copy(c) for c in snap.clauses],
        atoms=[copy.copy(a) for a in snap.theory_atoms],
        numerics=[copy.copy(n) for n in snap.numeric_vars],
        objective=copy.copy(snap.objective),
    )
