"""合成 OMT(LIA) 求解器快照生成器。

用于离线构造训练/测试数据。每个 instance 模拟一个 CDCL(T)+OMT 求解器在
某 decision/branch 点的局部状态，包含：
- 布尔变量与 CNF 子句
- 理论原子（线性不等式）
- 整数变量与其 LP 松弛值
- 目标函数与全局搜索状态
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Hashable

from omt_branching.input.solver_state import (
    BooleanVarInfo,
    ClauseInfo,
    NumericVarInfo,
    ObjectiveInfo,
    SearchStateInfo,
    SolverSnapshot,
    TheoryAtomInfo,
)
from omt_branching.interfaces import AtomKind, ClauseKind, SearchMode


@dataclass
class SyntheticConfig:
    """控制合成 instance 的规模与难度。"""

    n_bool: int = 20           # 布尔变量数
    n_numeric: int = 10        # 整数变量数
    n_clauses: int = 40        # CNF 子句数
    n_atoms: int = 15          # 理论原子数
    clause_width: int = 3      # 每个子句平均文字数
    atom_width: int = 3        # 每个理论原子平均涉及数值变量数
    soft_ratio: float = 0.15   # 多少布尔变量来自 soft constraint
    seed: int | None = None


def _rand_coeff(rng: random.Random, nonzero: bool = True) -> float:
    if not nonzero:
        return 0.0
    return rng.choice([-1, -1, 1, 1, 2, -2]) * rng.uniform(0.5, 2.0)


def generate_snapshot(cfg: SyntheticConfig = SyntheticConfig()) -> SolverSnapshot:
    """生成一个合成 OMT 快照。"""
    rng = random.Random(cfg.seed)

    bool_ids = [f"b{i}" for i in range(cfg.n_bool)]
    num_ids = [f"x{i}" for i in range(cfg.n_numeric)]

    # ---- 布尔变量 ----
    bool_vars: list[BooleanVarInfo] = []
    for i, vid in enumerate(bool_ids):
        is_soft = rng.random() < cfg.soft_ratio
        # 模拟一部分已赋值
        assigned = rng.random() < 0.25 and i < cfg.n_bool * 0.4
        assignment = None
        decision_level = None
        if assigned:
            assignment = rng.choice([True, False])
            decision_level = rng.randint(1, max(1, cfg.n_bool // 5))
        # VSIDS activity 与变量出现次数正相关（带噪声），使其成为有意义信号
        occurrence = rng.randint(1, max(2, cfg.n_clauses // 2))
        bool_vars.append(
            BooleanVarInfo(
                var_id=vid,
                assignment=assignment,
                decision_level=decision_level,
                is_candidate=(not assigned),
                vsids_activity=math.log1p(occurrence) * rng.uniform(0.5, 2.0),
                lrb_score=rng.random() * 0.8,
                chb_score=rng.random() * 0.6,
                phase_saved=rng.choice([True, False, None]),
                occurrence_count=occurrence,
                pos_count=rng.randint(0, 5),
                neg_count=rng.randint(0, 5),
                is_soft=is_soft,
                in_recent_learned=rng.random() < 0.2,
            )
        )

    # ---- CNF 子句 ----
    clauses: list[ClauseInfo] = []
    for i in range(cfg.n_clauses):
        width = max(2, rng.randint(cfg.clause_width - 1, cfg.clause_width + 1))
        lits = []
        used = set()
        while len(lits) < width:
            vid = rng.choice(bool_ids)
            if vid in used:
                continue
            used.add(vid)
            lits.append((vid, rng.choice([True, False])))
        clauses.append(
            ClauseInfo(
                clause_id=f"c{i}",
                literals=lits,
                kind=rng.choice([ClauseKind.ORIGINAL, ClauseKind.LEARNED]),
                lbd=rng.randint(1, width),
                activity=rng.random(),
                is_satisfied=None,
            )
        )

    # ---- 数值变量 ----
    numeric_vars: list[NumericVarInfo] = []
    lp_values: dict[Hashable, float] = {}
    for vid in num_ids:
        lb = rng.uniform(0.0, 5.0)
        ub = lb + rng.uniform(5.0, 15.0)
        lp = rng.uniform(lb, ub)
        lp_values[vid] = lp
        numeric_vars.append(
            NumericVarInfo(
                num_var_id=vid,
                is_integer=True,
                lp_value=lp,
                lower_bound=lb,
                upper_bound=ub,
                is_fractional=abs(lp - round(lp)) > 1e-3,
                objective_coeff=rng.choice([-3, -2, -1, 1, 2, 3]) * rng.uniform(0.5, 2.0),
                reduced_cost=rng.uniform(-1.0, 1.0),
                pseudocost_up=rng.uniform(0.1, 2.0),
                pseudocost_down=rng.uniform(0.1, 2.0),
            )
        )

    # ---- 理论原子：线性不等式 a^T x <= b / >= b ----
    atoms: list[TheoryAtomInfo] = []
    for i in range(cfg.n_atoms):
        bool_var = bool_ids[i % cfg.n_bool]
        width = max(1, rng.randint(cfg.atom_width - 1, cfg.atom_width + 1))
        coeffs: dict[Hashable, float] = {}
        for _ in range(width):
            vid = rng.choice(num_ids)
            coeffs[vid] = coeffs.get(vid, 0.0) + _rand_coeff(rng)
        rhs = rng.uniform(-5.0, 10.0)
        kind = rng.choice([AtomKind.LE, AtomKind.GE, AtomKind.EQ])
        # 计算 LP 下的 LHS
        lhs = sum(coeffs.get(v, 0.0) * lp_values[v] for v in coeffs)
        slack = rhs - lhs if kind in (AtomKind.LE, AtomKind.EQ) else lhs - rhs
        atoms.append(
            TheoryAtomInfo(
                atom_id=f"a{i}",
                bool_var_id=bool_var,
                kind=kind,
                var_coeffs=coeffs,
                rhs=rhs,
                lp_value=lhs,
                slack=slack,
                violation=max(0.0, -slack) if kind != AtomKind.EQ else abs(slack),
                tightens_objective=rng.random() < 0.3,
            )
        )

    # ---- 目标函数 ----
    obj_coeffs = {vid: nv.objective_coeff for vid, nv in zip(num_ids, numeric_vars)}
    incumbent = sum(obj_coeffs[v] * lp_values[v] for v in num_ids) + rng.uniform(-2.0, 2.0)
    best_bound = incumbent - rng.uniform(0.5, 5.0)
    soft_weights = {
        bv.var_id: rng.uniform(1.0, 5.0)
        for bv in bool_vars
        if bv.is_soft and bv.var_id is not None
    }
    objective = ObjectiveInfo(
        sense_is_min=True,
        incumbent=incumbent,
        best_bound=best_bound,
        gap=abs(incumbent - best_bound) / (abs(incumbent) + 1e-6),
        var_coeffs=obj_coeffs,
        soft_weights=soft_weights,
        related_bounds={vid: (nv.lower_bound, nv.upper_bound) for vid, nv in zip(num_ids, numeric_vars)},
    )

    # ---- 搜索状态 ----
    state = SearchStateInfo(
        depth=rng.randint(1, 20),
        decision_level=rng.randint(1, 10),
        trail_length=rng.randint(5, 50),
        conflict_count=rng.randint(0, 100),
        search_mode=rng.choice(list(SearchMode)),
        time_budget_left=rng.uniform(10.0, 300.0),
    )

    return SolverSnapshot(
        bool_vars=bool_vars,
        clauses=clauses,
        theory_atoms=atoms,
        numeric_vars=numeric_vars,
        objective=objective,
        search_state=state,
        snapshot_id=f"syn-{cfg.n_bool}-{cfg.n_numeric}-{rng.randint(0, 1_000_000)}",
    )


def generate_dataset(n: int, cfg: SyntheticConfig = SyntheticConfig()) -> list[SolverSnapshot]:
    """生成 n 个不重复种子的合成快照。"""
    return [generate_snapshot(SyntheticConfig(**{**cfg.__dict__, "seed": cfg.seed + i if cfg.seed is not None else i})) for i in range(n)]
