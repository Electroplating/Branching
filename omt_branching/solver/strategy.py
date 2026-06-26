"""分支策略：Neural（advice -> split）与 Baseline（输出半边桥接缝）。

``NeuralStrategy`` 把 ``BranchingAdvice`` 翻译成 F-Split 的子公式：对 LIA 优先做
**数值域切分**（B&B 风格，对应整数 head），无可用数值候选时退回布尔原子切分；
``use_gnn`` 为假或低置信时退回 ``resolve``（linear search），从而即便策略未训练，
GOMT 的 soundness 也保证最优（Thm 1）。

为使切分有意义，抽取时把当前 branch ``ψ`` 并入 ``φ`` 视图（``φ∧ψ``），使数值变量
的上下界随下降而收紧，切分点随之变化（否则会反复在同一点切分）。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Optional

from omt_branching.service import BranchingPolicyService
from omt_branching.solver.extractor import Handle, Z3SnapshotExtractor
from omt_branching.solver.interfaces import SplitDecision


@dataclass(frozen=True)
class StrategyConfig:
    """策略配置。``max_split_depth`` 限制每个 Δ-round 的 split 预算（保证终止）。"""

    max_split_depth: int = 6
    min_confidence: float = 0.0


def _numeric_split(handle: Handle, psi, backend, branch_up: bool) -> Optional[list]:
    """对数值变量做域切分 ``x ≤ m`` ∨ ``x ≥ m+1``；不可分返回 ``None``。"""
    lo, up = handle.lower, handle.upper
    if lo is None or up is None or up - lo < 1:
        return None
    m = int((lo + up) // 2)
    m = max(int(lo), min(m, int(up) - 1))
    if m < lo or m + 1 > up:
        return None
    low_branch = backend.conjoin(psi, backend.le(handle.z3_obj, m))
    high_branch = backend.conjoin(psi, backend.ge(handle.z3_obj, m + 1))
    return [high_branch, low_branch] if branch_up else [low_branch, high_branch]


def _bool_split(handle: Handle, psi, backend, phase_true: bool) -> list:
    """对布尔原子做 ``ψ∧a`` ∨ ``ψ∧¬a`` 切分；``phase_true`` 决定先探哪侧。"""
    atom = handle.z3_obj
    true_branch = backend.conjoin(psi, atom)
    false_branch = backend.conjoin(psi, backend.negate(atom))
    return [true_branch, false_branch] if phase_true else [false_branch, true_branch]


class NeuralStrategy:
    """用 Neural 策略驱动 F-Split。"""

    def __init__(self, problem, service: BranchingPolicyService,
                 config: StrategyConfig = StrategyConfig()):
        self.problem = problem
        self.service = service
        self.config = config
        self.extractor = Z3SnapshotExtractor(problem)

    def propose(self, state, backend) -> SplitDecision:
        depth = state.stats.get("branch_depth", 0)
        if depth >= self.config.max_split_depth:
            return SplitDecision.resolve()
        try:
            view = replace(state, hard=backend.conjoin(state.hard, state.top))
            extraction = self.extractor.extract(view, backend)
            advice = self.service.advise(extraction.snapshot)
        except Exception as exc:  # 任何抽取/推理异常都不应中断搜索
            warnings.warn(f"NeuralStrategy 抽取/推理失败，回退 resolve: {exc!r}")
            return SplitDecision.resolve()

        if not advice.use_gnn or advice.confidence < self.config.min_confidence:
            return SplitDecision.resolve()

        psi = state.top
        subs: Optional[list] = None

        # 主路径：数值域切分（LIA 的 B&B 分支）。
        split_advice = advice.integer_split
        if split_advice is not None:
            handle = extraction.numeric_handles.get(split_advice.num_var_id)
            if handle is not None:
                subs = _numeric_split(handle, psi, backend, split_advice.branch_up)

        # 退回：布尔原子切分。
        if subs is None:
            candidate = advice.top_candidate()
            if candidate is not None:
                handle = extraction.atom_handles.get(candidate)
                if handle is not None:
                    phase_true = advice.phase_suggestions.get(candidate, True)
                    subs = _bool_split(handle, psi, backend, phase_true)

        if subs is None:
            return SplitDecision.resolve()
        state.stats["branch_depth"] = depth + 1
        return SplitDecision.split(subs, source="neural")


class BaselineStrategy:
    """非神经基线。

    - 给定 ``problem``：复用抽取器做**最大域变量的中点二分**（确定性启发式）。
    - 不给 ``problem``：退化为 linear search（始终 ``resolve``，GOMT §4.2）。
    两种配置都满足 soundness（与 NeuralStrategy 同一 calculus）。
    """

    def __init__(self, problem=None, config: StrategyConfig = StrategyConfig()):
        self.config = config
        self.extractor = Z3SnapshotExtractor(problem) if problem is not None else None

    def propose(self, state, backend) -> SplitDecision:
        depth = state.stats.get("branch_depth", 0)
        if self.extractor is None or depth >= self.config.max_split_depth:
            return SplitDecision.resolve()
        try:
            view = replace(state, hard=backend.conjoin(state.hard, state.top))
            extraction = self.extractor.extract(view, backend)
        except Exception as exc:
            warnings.warn(f"BaselineStrategy 抽取失败，回退 resolve: {exc!r}")
            return SplitDecision.resolve()

        best: Optional[Handle] = None
        best_span = 0.0
        for handle in extraction.numeric_handles.values():
            if handle.lower is None or handle.upper is None:
                continue
            span = handle.upper - handle.lower
            if span >= 1 and span > best_span:
                best, best_span = handle, span
        if best is None:
            return SplitDecision.resolve()

        subs = _numeric_split(best, state.top, backend, branch_up=False)
        if subs is None:
            return SplitDecision.resolve()
        state.stats["branch_depth"] = depth + 1
        return SplitDecision.split(subs, source="baseline")


__all__ = ["NeuralStrategy", "BaselineStrategy", "StrategyConfig"]
