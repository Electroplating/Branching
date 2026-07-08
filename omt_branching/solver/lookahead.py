"""SAT look-ahead 布尔专家（Phase 2 imitation 冷启动标签）。

对布尔骨架 φ 的每个候选原子 a 做**一层前瞻**：分别试探性赋 a=True / a=False，用 z3 的
``Solver.consequences`` 计算**单元传播/蕴含**闭包的规模，据此给原子打分——两侧都能传播出大量
其它文字的原子是"好决策"（经典 DPLL look-ahead / March 启发式），恰有一侧 UNSAT 的原子被 φ
蕴含（等价单元）故记大分（sentinel），phase 取被强制/传播更多的一侧先探。

产出直接喂给现有 :class:`~omt_branching.model.trainer.ImitationTrainer`（bool head + phase
head）。与 GOMT F-Split 路径的 ``strong_branch`` 专家解耦：这里只作用于 propagator/decide 路径
的**布尔骨架**（原子 + 子句共现图，见 spec §5），不涉及目标分离度与数值域切分。

所有 z3 交互仅用稳定 API（``Solver.check`` / ``Solver.consequences``），不改 z3。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional

import z3

from omt_branching.input.graph_builder import DEFAULT_FEATURE_SPEC, FeatureSpec, GraphBuilder
from omt_branching.interfaces import NodeType
from omt_branching.model.policy import BranchingPolicy
from omt_branching.model.trainer import RankingExample
from omt_branching.solver.propagator_snapshot import (
    atom_key, build_bool_snapshot, collect_atoms,
)


@dataclass(frozen=True)
class LookaheadConfig:
    """look-ahead 专家配置。

    - ``max_atoms``：每实例前瞻的候选原子上限（前瞻昂贵，可子采样）。
    - ``product_weight``：两侧传播数乘积项权重（偏好两侧都能传播的"平衡"原子）。
    - ``sentinel``：恰一侧 UNSAT（原子被 φ 蕴含=等价单元）时记的大分。
    - ``eps``：分数下限（判定"无有意义分支"）。
    """

    max_atoms: int = 32
    product_weight: float = 1024.0
    sentinel: float = 1_000_000.0
    eps: float = 1e-9


def _propagation_count(solver, lit, probe) -> Optional[int]:
    """在 ``solver`` 上试探赋 ``lit`` 为真，返回 ``probe`` 中被蕴含的文字数；该侧 UNSAT 返回 None。"""
    try:
        res, implied = solver.consequences([lit], probe)
    except z3.Z3Exception:
        return 0
    if res == z3.unsat:
        return None
    return len(implied)


def lookahead_scores(assertions, atoms: Optional[list] = None,
                     config: LookaheadConfig = LookaheadConfig()):
    """对布尔骨架的候选原子做一层前瞻打分。

    返回 ``(scores, phases)``，均以 :func:`atom_key` 为键：

    - ``scores[k]``：两侧可行时 ``w·p⁺·p⁻ + p⁺ + p⁻``（``p±`` 为赋真/假后的蕴含文字数）；
      恰一侧 UNSAT 记 ``sentinel``（等价单元，极强）。
    - ``phases[k]``：先探"传播更多/被强制"的一侧（UNSAT 侧的反面 = 可行/强制值）。
    """
    atoms = atoms if atoms is not None else collect_atoms(list(assertions))
    solver = z3.Solver()
    solver.add(*assertions)
    if solver.check() != z3.sat:            # φ 本身 UNSAT：无从前瞻
        return {}, {}

    probe = list(atoms)                     # 蕴含闭包在全体原子上度量
    scores: dict[Hashable, float] = {}
    phases: dict[Hashable, bool] = {}
    for a in atoms[: config.max_atoms]:
        k = atom_key(a)
        p_pos = _propagation_count(solver, a, probe)
        p_neg = _propagation_count(solver, z3.Not(a), probe)
        if p_pos is None and p_neg is None:
            continue                        # 两侧皆 UNSAT（φ 已 UNSAT，不应到此）
        if p_pos is None:                   # a 必假（a=真 -> UNSAT）
            scores[k] = config.sentinel
            phases[k] = False
            continue
        if p_neg is None:                   # a 必真（a=假 -> UNSAT）
            scores[k] = config.sentinel
            phases[k] = True
            continue
        scores[k] = config.product_weight * p_pos * p_neg + p_pos + p_neg
        phases[k] = p_pos >= p_neg          # 先探传播更多的一侧
    return scores, phases


def build_lookahead_example(assertions,
                            feature_spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
                            config: LookaheadConfig = LookaheadConfig()) -> Optional[RankingExample]:
    """把布尔骨架 + look-ahead 标签打包成 :class:`RankingExample`（bool head + phase head）。

    标签以**图内局部索引**给出（经 ``graph.id_maps[BOOL_VAR]``，键=原子 :func:`atom_key`）。
    无有意义分支（专家空/全 0）时返回 ``None``。
    """
    snap, amap = build_bool_snapshot(list(assertions))
    graph = GraphBuilder(feature_spec).build(snap)
    scores, phases = lookahead_scores(list(assertions), atoms=list(amap.values()), config=config)
    if not scores or max(scores.values()) <= config.eps:
        return None

    bmap = graph.id_maps.get(NodeType.BOOL_VAR, {})
    bool_scores: dict[int, float] = {}
    phase_targets: dict[int, bool] = {}
    for k, sc in scores.items():
        local = bmap.get(k)
        if local is not None:
            bool_scores[local] = sc
    for k, ph in phases.items():
        local = bmap.get(k)
        if local is not None:
            phase_targets[local] = ph
    if not bool_scores:
        return None
    return RankingExample(graph=graph, bool_target_scores=bool_scores,
                          phase_targets=phase_targets)


def build_lookahead_examples(assertion_lists,
                             feature_spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
                             config: LookaheadConfig = LookaheadConfig()) -> list[RankingExample]:
    """对一组布尔骨架批量构造 look-ahead imitation 样本（跳过无有意义分支者）。"""
    out: list[RankingExample] = []
    for asserts in assertion_lists:
        ex = build_lookahead_example(asserts, feature_spec, config)
        if ex is not None:
            out.append(ex)
    return out


def decide_bool_hit(policy: BranchingPolicy, assertions,
                    feature_spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
                    config: LookaheadConfig = LookaheadConfig()) -> Optional[bool]:
    """在同一布尔骨架上比较策略 bool-head top-1 与 look-ahead 专家 top-1 是否一致。

    无有意义分支或无 bool 候选返回 ``None``（该实例不计入准确率）。单次建图保证专家与策略
    共享同一套原子键，规避 id 漂移。
    """
    import torch

    snap, amap = build_bool_snapshot(list(assertions))
    graph = GraphBuilder(feature_spec).build(snap)
    scores, _ = lookahead_scores(list(assertions), atoms=list(amap.values()), config=config)
    if not scores or max(scores.values()) <= config.eps:
        return None
    oracle_key = max(scores, key=lambda k: scores[k])

    out = policy.infer(graph)
    probs = out.masked_bool_probs()
    if probs.numel() == 0 or not out.candidate_bool_local:
        return None
    local = int(torch.argmax(probs).item())
    return graph.solver_id(NodeType.BOOL_VAR, local) == oracle_key


__all__ = [
    "LookaheadConfig",
    "lookahead_scores",
    "build_lookahead_example",
    "build_lookahead_examples",
    "decide_bool_hit",
]
