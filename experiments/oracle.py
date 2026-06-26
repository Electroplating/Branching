"""专家标签生成器（强 heuristic / 简化 strong branching）。

为每个合成快照生成：
- 布尔候选的 branching 分数
- phase 标签
- 整数 B&B 分数与方向
- 辅助任务标签

这些标签作为 imitation learning 的监督信号，并与训练后的 GNN 做对比。
"""

from __future__ import annotations

import math
from typing import Hashable

from omt_branching.graph.hetero_graph import HeteroGraph
from omt_branching.input.graph_builder import GraphBuilder
from omt_branching.input.solver_state import SolverSnapshot
from omt_branching.interfaces import NodeType
from omt_branching.model.trainer import RankingExample


class OracleBrancher:
    """基于若干 OMT-aware heuristic 构造专家 score。

    综合规则（可视为廉价版 strong branching）：
    1. VSIDS / LRB activity 越高越优先。
    2. 出现在 tight / violated 理论原子中的变量优先。
    3. 与 objective 系数大的数值变量相关联的布尔变量优先。
    4. soft clause 权重高者优先（MaxSMT/OMT+PB 场景）。
    5. 频繁出现在子句中且未赋值的变量优先。
    """

    def __init__(self, weights: dict[str, float] | None = None):
        self.w = weights or {
            "vsids": 1.0,
            "lrb": 0.5,
            "theory": 2.0,
            "objective": 1.5,
            "soft": 1.0,
            "occurrence": 0.3,
            "recent_learned": 0.4,
        }

    def bool_branch_scores(self, snap: SolverSnapshot) -> dict[Hashable, float]:
        """返回 var_id -> 专家偏好分数（越大越该分支）。

        返回的分数已经过 softmax 归一化，可直接作为 imitation ranking 的软标签。
        """
        scores: dict[Hashable, float] = {}

        # 预计算每个数值变量对 objective 的影响强度
        obj_mag = {
            nv.num_var_id: abs(nv.objective_coeff) * (1.0 if nv.is_fractional else 0.5)
            for nv in snap.numeric_vars
        }

        # 预计算每个布尔变量关联的理论原子“紧张程度”
        atom_tightness: dict[Hashable, float] = {}
        for atom in snap.theory_atoms:
            tight = 0.0
            if atom.slack is not None:
                # slack 接近 0 表示紧张；violation 表示冲突
                tight = 1.0 / (1.0 + abs(atom.slack)) + (atom.violation or 0.0)
            if atom.tightens_objective:
                tight += 1.0
            atom_tightness[atom.bool_var_id] = atom_tightness.get(atom.bool_var_id, 0.0) + tight

        for bv in snap.bool_vars:
            if not bv.is_candidate or bv.is_eliminated or bv.assignment is not None:
                continue
            s = 0.0
            s += self.w["vsids"] * bv.vsids_activity
            s += self.w["lrb"] * bv.lrb_score
            s += self.w["theory"] * atom_tightness.get(bv.var_id, 0.0)
            s += self.w["occurrence"] * math.log1p(bv.occurrence_count)
            s += self.w["recent_learned"] * (1.0 if bv.in_recent_learned else 0.0)
            if bv.is_soft and bv.var_id in snap.objective.soft_weights:
                s += self.w["soft"] * snap.objective.soft_weights[bv.var_id]
            # 如果该变量对应的理论原子涉及高 objective 系数数值变量，额外加分
            for atom in snap.theory_atoms:
                if atom.bool_var_id == bv.var_id:
                    obj_link = sum(obj_mag.get(v, 0.0) * abs(c) for v, c in atom.var_coeffs.items())
                    s += self.w["objective"] * obj_link
            scores[bv.var_id] = s

        # softmax 归一化到概率分布（温度 T=1）
        if scores:
            max_s = max(scores.values())
            exp = {k: math.exp(v - max_s) for k, v in scores.items()}
            z = sum(exp.values())
            scores = {k: v / z for k, v in exp.items()}
        return scores

    def phase_labels(self, snap: SolverSnapshot) -> dict[Hashable, bool]:
        """phase 标签：优先满足 soft clause；否则使用 phase_saved。"""
        labels: dict[Hashable, bool] = {}
        for bv in snap.bool_vars:
            if bv.assignment is not None:
                continue
            if bv.is_soft:
                # 高权重 soft clause 优先满足（取真）
                labels[bv.var_id] = True
            elif bv.phase_saved is not None:
                labels[bv.var_id] = bv.phase_saved
            else:
                labels[bv.var_id] = True
        return labels

    def int_branch_scores(self, snap: SolverSnapshot) -> dict[Hashable, float]:
        """整数 B&B 分数：分数部分大、objective 系数大、pseudo-cost 高者优先。

        返回 softmax 归一化概率分布。
        """
        scores: dict[Hashable, float] = {}
        for nv in snap.numeric_vars:
            if not nv.is_fractional:
                continue
            frac = abs(nv.lp_value - round(nv.lp_value)) if nv.lp_value is not None else 0.0
            pc = (nv.pseudocost_up + nv.pseudocost_down) / 2.0
            scores[nv.num_var_id] = frac * (abs(nv.objective_coeff) + 0.1) * (pc + 0.1)
        if scores:
            max_s = max(scores.values())
            exp = {k: math.exp(v - max_s) for k, v in scores.items()}
            z = sum(exp.values())
            scores = {k: v / z for k, v in exp.items()}
        return scores

    def int_direction(self, snap: SolverSnapshot) -> dict[Hashable, bool]:
        """split 方向：最小化问题向上分支通常提高下界；这里简化用 objective 系数符号。"""
        dirs: dict[Hashable, bool] = {}
        for nv in snap.numeric_vars:
            if not nv.is_fractional or nv.lp_value is None:
                continue
            # 最小化：若系数为正，先向下可能更快获得好 incumbent；这里取向上探索
            dirs[nv.num_var_id] = nv.objective_coeff >= 0
        return dirs

    def make_example(self, snap: SolverSnapshot) -> RankingExample:
        """构造单个 RankingExample。"""
        builder = GraphBuilder()
        graph = builder.build(snap)

        bool_scores = self.bool_branch_scores(snap)
        phase = self.phase_labels(snap)
        int_scores = self.int_branch_scores(snap)
        int_dir = self.int_direction(snap)

        bool_map = graph.id_maps[NodeType.BOOL_VAR]
        num_map = graph.id_maps[NodeType.NUMERIC_VAR]

        # 只保留候选集合内的标签
        cand_bool = graph.meta.get("candidate_bool_ids", [])
        cand_num = graph.meta.get("candidate_numeric_ids", [])

        return RankingExample(
            graph=graph,
            bool_target_scores={bool_map[v]: s for v, s in bool_scores.items() if v in bool_map and v in cand_bool},
            phase_targets={bool_map[v]: p for v, p in phase.items() if v in bool_map},
            int_target_scores={num_map[v]: s for v, s in int_scores.items() if v in num_map and v in cand_num},
            int_dir_targets={num_map[v]: d for v, d in int_dir.items() if v in num_map and v in cand_num},
            conflict_targets={bool_map[v]: False for v in cand_bool if v in bool_map},  # 简化：未知
            obj_improve_targets={bool_map[v]: bool_scores.get(v, 0.0) for v in cand_bool if v in bool_map},
        )
