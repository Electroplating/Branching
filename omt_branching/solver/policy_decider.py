"""PolicyDecider：把 GNN 策略包成 propagator 的 decide 函数（周期 refocus）。

每 ``refocus_every`` 次决策重算一次原子优先级（调用 GNN，代价可控），decide 时 O(1) 取
最高优先级的未定原子。策略 ``use_gnn=False`` 时返回 None -> propagator 退回 VSIDS。

冲突回退时（propagator ``pop`` → :meth:`on_backtrack`）可立即强制下次 decide refocus，
使优先级与回退后的部分赋值 / 投影图一致。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from omt_branching.model.device import gnn_device
from omt_branching.model.inference import InferenceConfig
from omt_branching.model.persistence import load_policy
from omt_branching.model.policy import BranchingPolicy
from omt_branching.service import BranchingPolicyService, ServiceConfig
from omt_branching.solver.decide_omt import solve_omt_with_decider
from omt_branching.solver.interfaces import Sense
from omt_branching.solver.propagator_snapshot import build_bool_snapshot


class PolicyDecider:
    def __init__(
        self,
        service: BranchingPolicyService,
        assertions,
        refocus_every: int = 200,
        *,
        refocus_on_backtrack: bool = True,
    ):
        self.service = service
        self.assertions = list(assertions)
        self.refocus_every = max(1, refocus_every)
        self.refocus_on_backtrack = refocus_on_backtrack
        self._pri: Optional[dict] = None
        self._phase: dict = {}
        self._since = self.refocus_every   # 首次即 refocus

    def force_refocus(self) -> None:
        """清空缓存优先级，使下次 decide 立刻跑 GNN。"""
        self._pri = None
        self._since = self.refocus_every

    def add_hard(self, *exprs) -> None:
        """把新增硬约束（如 OMT better-cut）并入建图断言，并强制下次 decide refocus。"""
        if not exprs:
            return
        self.assertions.extend(exprs)
        self.force_refocus()

    def on_backtrack(self, num_scopes: int = 1) -> None:
        """propagator ``pop`` 回调：冲突回退后强制下次 decide refocus。"""
        if self.refocus_on_backtrack:
            self.force_refocus()

    def _refocus(self, assignment):
        snap, _ = build_bool_snapshot(self.assertions, assignment=assignment)
        try:
            advice = self.service.advise(snap)
        except Exception:
            self._pri = None
            return
        if not advice.use_gnn:
            self._pri = None
            return
        self._pri = dict(advice.activity_priors)
        self._phase = dict(advice.phase_suggestions)

    def __call__(self, undecided_keys, assignment) -> Optional[tuple]:
        if self._since >= self.refocus_every:
            self._refocus(assignment)
            self._since = 0
        self._since += 1
        if not self._pri:
            return None
        cand = [k for k in undecided_keys if k in self._pri]
        if not cand:
            return None
        best = max(cand, key=lambda k: self._pri[k])
        return best, bool(self._phase.get(best, True))


def solve_with_learned_policy(
    hard,
    objective,
    sense: Sense,
    *,
    checkpoint: Union[str, Path, None] = None,
    policy: BranchingPolicy | None = None,
    refocus_every: int = 50,
    device: str | None = None,
    max_iters: int = 100000,
) -> dict:
    """用 checkpoint / 已有 ``BranchingPolicy`` + :class:`PolicyDecider` 求解单实例。

    ``checkpoint`` 为 :func:`omt_branching.model.persistence.save_policy` 写出的 ``.pt``；
    与 ``policy`` 二选一（同时给时以 ``policy`` 为准）。便于调试 RL 中间/最终权重。
    """
    if policy is None:
        if checkpoint is None:
            raise ValueError("必须提供 checkpoint 或 policy")
        policy, _meta = load_policy(checkpoint, map_location="cpu")
    dev = device or gnn_device()
    policy = policy.to(dev)
    policy.eval()
    svc = BranchingPolicyService(
        policy=policy,
        config=ServiceConfig(inference=InferenceConfig(device=dev)),
    )
    return solve_omt_with_decider(
        hard,
        objective,
        sense,
        decider_factory=lambda a: PolicyDecider(svc, a, refocus_every),
        max_iters=max_iters,
    )


__all__ = ["PolicyDecider", "solve_with_learned_policy"]
