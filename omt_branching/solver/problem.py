"""GOMT 问题定义与初始状态构造（不依赖 z3）。

``GOMTProblem ⟨t, ≺, φ⟩`` 对应 ``GOMT.pdf`` Definition 1 的单目标特例：``φ`` 为硬
约束，``t`` 为目标项，``≺`` 由 ``Sense`` 给出（MIN/MAX）。``initial_state`` 实现
Definition 11：``I₀ = Solve(φ)``，``Δ₀ = Better(I₀)``，``τ₀ = (Δ₀)``。
"""

from __future__ import annotations

from dataclasses import dataclass

from omt_branching.solver.interfaces import (
    Constraint, GOMTState, Sense, SolveBackend, Term,
)


class Infeasible(Exception):
    """硬约束 ``φ`` 不可满足，优化无意义。"""


def _fresh_stats() -> dict:
    return {"splits": 0, "sats": 0, "closes": 0, "solve_calls": 0, "branch_depth": 0}


@dataclass(frozen=True)
class GOMTProblem:
    """单目标 GOMT 问题。

    - ``hard_list``：硬约束 ``φ`` 的合取项（``tuple``，由 backend 合取成单一公式）。
    - ``objective``：目标项 ``t``（后端不透明句柄）。
    - ``sense``：优化方向。
    """

    hard_list: tuple[Constraint, ...]
    objective: Term
    sense: Sense

    def initial_state(self, backend: SolveBackend) -> GOMTState:
        """构造初始状态（Definition 11）。``φ`` 不可满足时抛 :class:`Infeasible`。"""
        phi = backend.conjoin(*self.hard_list)
        incumbent = backend.solve(phi)
        if incumbent is None:
            raise Infeasible("硬约束 φ 不可满足")
        value0 = backend.value(incumbent, self.objective)
        delta0 = backend.better(self.objective, value0, self.sense)
        return GOMTState(
            incumbent=incumbent,
            delta=delta0,
            tau=[delta0],
            objective=self.objective,
            sense=self.sense,
            hard=phi,
            step=0,
            stats=_fresh_stats(),
        )


__all__ = ["GOMTProblem", "Infeasible"]
