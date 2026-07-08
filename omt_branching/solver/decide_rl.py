"""Solver-in-the-Loop 强化学习：在 z3 内部布尔决策回路里用 REINFORCE 微调 GNN 分支策略
（Phase 2）。

与 GOMT F-Split 的 :mod:`omt_branching.solver.rl` 不同，这里的策略经 **UserPropagator /
decide** 直接接管 z3 内部布尔决策（spec §3.6）：

- **动作**：每次 **refocus** 时策略在候选原子上给出优先级分布，从中**采样**一个"聚焦原子"
  （及其相位）——该原子随后被优先 ``next_split``，从而直接驱动 z3 的内部决策。
- **奖励**：纯终局奖励 = ``-log(1+rlimit)``（整体 OMT 求解消耗的 z3 rlimit count）。OMT
  线性搜索回路本身保证到达 ``== native`` 最优（正确性与分支无关），故优化目标就是"用更少
  的求解开销到达最优"。
- **算法**：REINFORCE，带 per-instance 移动平均 baseline 与熵正则；imitation（look-ahead）
  冷启动后微调。

rollout 时 ``policy.infer`` 无梯度采样并记录每步的图与所选局部索引；更新时对记录的每个图重跑
一次前向（图为固定输入，梯度只流向策略参数）。
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch

from omt_branching.graph.hetero_graph import HeteroGraph
from omt_branching.input.graph_builder import DEFAULT_FEATURE_SPEC, FeatureSpec, GraphBuilder
from omt_branching.interfaces import NodeType
from omt_branching.model.policy import BranchingPolicy, _masked_softmax
from omt_branching.service import BranchingPolicyService
from omt_branching.solver.decide_omt import solve_omt_with_decider
from omt_branching.solver.interfaces import Sense
from omt_branching.solver.propagator_snapshot import build_bool_snapshot


# --------------------------------------------------------------------------- #
# 配置与轨迹数据结构
# --------------------------------------------------------------------------- #
@dataclass
class DecideRLConfig:
    """decide 路径 RL 配置。"""

    lr: float = 3e-4
    gamma: float = 0.99               # 折扣因子（本设定为纯终局奖励，主要保持接口一致）
    entropy_coef: float = 1e-2        # 熵正则权重（鼓励探索）
    baseline_momentum: float = 0.9    # per-instance 移动平均 baseline 动量
    grad_clip: float = 5.0
    device: str = "cpu"

    refocus_every: int = 200          # 每多少次 decide 重算一次优先级（= 一个 RL 动作）
    max_record: int = 512             # 单 episode 记录动作数上限（控制显存/算力）
    rlimit_penalty_coef: float = 1.0  # 终局 rlimit 代价 penalty 权重
    use_log_cost: bool = True         # True: 用 log(1+rlimit) 压缩代价尺度
    max_iters: int = 100_000          # OMT 线性搜索回路的迭代上限
    eps: float = 1e-9


@dataclass
class DecideStep:
    """一次被记录的 refocus 动作（一个 RL 动作）。"""

    graph: HeteroGraph                # 该 refocus 点的图（供更新时重跑前向）
    cand_locals: list[int]            # 候选布尔原子的图内局部索引
    chosen_local: int                 # 采样到的"聚焦原子"局部索引
    phase: bool                       # 采样到的相位（取真=True）


@dataclass
class DecideEpisode:
    """一次完整 OMT 求解产生的动作轨迹与终局奖励。"""

    steps: list[DecideStep] = field(default_factory=list)
    terminal_reward: float = 0.0
    rlimit: int = 0
    conflicts: int = 0
    decisions: Optional[int] = None
    value: object = None
    runtime: float = 0.0


# --------------------------------------------------------------------------- #
# 采样式 decider：在 refocus 处采样聚焦原子并记录
# --------------------------------------------------------------------------- #
class SamplingDecider:
    """把 GNN 策略包成 propagator 的 decide 函数，并在每次 refocus 采样一个"聚焦原子"。

    - ``sample=True``（训练）：按候选原子上的策略分布采样聚焦原子与相位，产生 on-policy 轨迹。
    - ``sample=False``（评估）：取 argmax，等价确定性优先级（与部署一致）。

    每次 refocus 把 ``(graph, cand_locals, chosen_local, phase)`` 追加到 ``self.steps``，供
    :class:`DecideRLTrainer` 计算奖励与梯度。策略无候选/推理异常时返回 ``None`` -> 退回 VSIDS。
    """

    def __init__(self, policy: BranchingPolicy, assertions,
                 config: DecideRLConfig = DecideRLConfig(),
                 feature_spec: FeatureSpec = DEFAULT_FEATURE_SPEC, sample: bool = True):
        self.policy = policy
        self.assertions = list(assertions)
        self.config = config
        self.builder = GraphBuilder(feature_spec)
        self.sample = sample
        self.steps: list[DecideStep] = []
        self._scores: dict = {}         # key -> float 优先级（候选原子）
        self._phase: dict = {}          # key -> bool 相位
        self._focus_key = None          # 本轮聚焦原子键
        self._since = config.refocus_every   # 首次即 refocus

    # ------------------------------------------------------------------ #
    def _refocus(self, assignment) -> None:
        self._scores, self._phase, self._focus_key = {}, {}, None
        try:
            snap, _ = build_bool_snapshot(self.assertions, assignment=assignment)
            graph = self.builder.build(snap)
            out = self.policy.infer(graph)
        except Exception:
            return
        cand = list(out.candidate_bool_local)
        probs = _masked_softmax(out.bool_branch_scores, cand)
        if probs.numel() == 0 or not cand or float(probs.sum()) <= 0:
            return

        # 记录动作用的分数/相位（全体候选，供 decide 排序与更新重算）。
        for local in cand:
            key = graph.solver_id(NodeType.BOOL_VAR, local)
            if key is None:
                continue
            self._scores[key] = float(out.bool_branch_scores[local])
            self._phase[key] = float(torch.sigmoid(out.phase_logits[local])) >= 0.5

        focus_local = self._pick(probs)
        phase_p = float(torch.sigmoid(out.phase_logits[focus_local]))
        focus_phase = (float(torch.rand(())) < phase_p) if self.sample else (phase_p >= 0.5)
        self._focus_key = graph.solver_id(NodeType.BOOL_VAR, focus_local)
        self._phase[self._focus_key] = focus_phase

        if len(self.steps) < self.config.max_record:
            self.steps.append(DecideStep(graph=graph, cand_locals=cand,
                                         chosen_local=focus_local, phase=focus_phase))

    def _pick(self, probs: torch.Tensor) -> int:
        if self.sample:
            return int(torch.multinomial(probs, 1).item())
        return int(torch.argmax(probs).item())

    # ------------------------------------------------------------------ #
    def __call__(self, undecided_keys, assignment) -> Optional[tuple]:
        if self._since >= self.config.refocus_every:
            self._refocus(assignment)
            self._since = 0
        self._since += 1
        if not self._scores:
            return None
        # 聚焦原子未定则优先探它（采样动作直接驱动决策）；否则取当前最高优先级未定原子。
        if self._focus_key is not None and self._focus_key in undecided_keys:
            return self._focus_key, bool(self._phase.get(self._focus_key, True))
        cand = [k for k in undecided_keys if k in self._scores]
        if not cand:
            return None
        best = max(cand, key=lambda k: self._scores[k])
        return best, bool(self._phase.get(best, True))


# --------------------------------------------------------------------------- #
# 强化学习训练器
# --------------------------------------------------------------------------- #
class DecideRLTrainer:
    """在 z3 内部布尔决策回路中用 REINFORCE 微调分支策略（decide 路径）。"""

    def __init__(self, policy: BranchingPolicy, config: DecideRLConfig = DecideRLConfig()):
        self.policy = policy.to(config.device)
        self.config = config
        self.opt = torch.optim.Adam(policy.parameters(), lr=config.lr)
        self._baseline = 0.0
        self._baselines: dict = {}      # 实例键 -> 移动平均 baseline

    # ------------------------------------------------------------------ #
    # 采集：跑一次真实 OMT 求解，记录 refocus 动作并算终局奖励
    # ------------------------------------------------------------------ #
    def collect_episode(self, hard, objective, sense: Sense) -> DecideEpisode:
        decider = SamplingDecider(self.policy, list(hard), self.config, sample=True)
        t0 = time.perf_counter()
        res = solve_omt_with_decider(hard, objective, sense,
                                     decider_factory=lambda a: decider,
                                     max_iters=self.config.max_iters)
        runtime = time.perf_counter() - t0

        rlimit = int(res.get("rlimit", 0) or 0)
        cost = math.log1p(rlimit) if self.config.use_log_cost else float(rlimit)
        terminal = -self.config.rlimit_penalty_coef * cost
        return DecideEpisode(
            steps=decider.steps, terminal_reward=terminal, rlimit=rlimit,
            conflicts=int(res.get("conflicts", 0) or 0), decisions=res.get("decisions"),
            value=res.get("value"), runtime=runtime,
        )

    # ------------------------------------------------------------------ #
    # baseline
    # ------------------------------------------------------------------ #
    def _baseline_for(self, key) -> float:
        if key is None:
            return self._baseline
        return self._baselines.get(key, self._baseline)

    def _update_baseline_for(self, key, value: float) -> None:
        m = self.config.baseline_momentum
        if key is None:
            self._baseline = m * self._baseline + (1 - m) * value
        elif key in self._baselines:
            self._baselines[key] = m * self._baselines[key] + (1 - m) * value
        else:
            self._baselines[key] = value

    # ------------------------------------------------------------------ #
    # 更新：REINFORCE（纯终局奖励，所有动作共享同一回报）
    # ------------------------------------------------------------------ #
    def update(self, episode: DecideEpisode, key=None) -> dict[str, float]:
        G = episode.terminal_reward
        if not episode.steps:
            return {"loss": 0.0, "return": G, "baseline": self._baseline_for(key), "steps": 0}

        self.policy.train()
        dev = self.config.device
        advantage = G - self._baseline_for(key)

        policy_loss = torch.zeros((), device=dev)
        entropy = torch.zeros((), device=dev)
        n = 0
        for step in episode.steps:
            g = step.graph.to(dev)
            out = self.policy(g)
            logp, ent = self._action_logp_entropy(out, step)
            if logp is None:
                continue
            policy_loss = policy_loss - logp * advantage
            entropy = entropy + ent
            n += 1

        if n == 0:
            return {"loss": 0.0, "return": G, "baseline": self._baseline_for(key), "steps": 0}

        loss = (policy_loss - self.config.entropy_coef * entropy) / n
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
        self.opt.step()

        self._update_baseline_for(key, G)
        return {"loss": float(loss), "return": G,
                "baseline": self._baseline_for(key), "steps": n}

    def _action_logp_entropy(self, out, step: DecideStep):
        """重算所记录 refocus 动作（聚焦原子选择 + 相位）的对数似然与分布熵。"""
        probs = _masked_softmax(out.bool_branch_scores, step.cand_locals)
        if probs.numel() == 0 or step.chosen_local >= probs.numel():
            return None, None
        p_sel = probs[step.chosen_local].clamp_min(1e-12)
        dir_p = torch.sigmoid(out.phase_logits[step.chosen_local]).clamp(1e-6, 1 - 1e-6)
        p_dir = dir_p if step.phase else (1.0 - dir_p)
        logp = torch.log(p_sel) + torch.log(p_dir)
        ent = _categorical_entropy(probs) + _bernoulli_entropy(dir_p)
        return logp, ent

    # ------------------------------------------------------------------ #
    # 训练主循环 & 评估
    # ------------------------------------------------------------------ #
    def train(self, instances, iterations: int = 1, log: bool = True) -> list[dict[str, float]]:
        """对一组 ``(hard, objective, sense)`` 反复采集 + 更新。"""
        instances = list(instances)
        history: list[dict[str, float]] = []
        for it in range(iterations):
            for j, (hard, obj, sense) in enumerate(instances):
                ep = self.collect_episode(hard, obj, sense)
                stats = self.update(ep, key=j)
                stats.update({"iter": it, "instance": j, "rlimit": ep.rlimit,
                              "decisions": ep.decisions, "value": ep.value,
                              "runtime": ep.runtime})
                history.append(stats)
                if log:
                    print(f"[iter {it} inst {j}] loss={stats['loss']:.4f} "
                          f"return={stats['return']:.4f} baseline={stats['baseline']:.4f} "
                          f"rlimit={ep.rlimit} decisions={ep.decisions} value={ep.value}")
        return history

    @torch.no_grad()
    def evaluate(self, hard, objective, sense: Sense) -> dict:
        """确定性评估（argmax 优先级），返回与 :func:`solve_omt_with_decider` 一致的指标 dict。"""
        decider = SamplingDecider(self.policy, list(hard), self.config, sample=False)
        return solve_omt_with_decider(hard, objective, sense,
                                      decider_factory=lambda a: decider,
                                      max_iters=self.config.max_iters)

    def make_service(self) -> BranchingPolicyService:
        """把当前策略包装成部署用的 :class:`BranchingPolicyService`。"""
        return BranchingPolicyService(policy=self.policy)

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #
    def save(self, path, history: Optional[list] = None) -> None:
        from omt_branching.model.persistence import save_policy

        save_policy(self.policy, path, meta={
            "kind": "decide_rl", "baseline": self._baseline,
            "rl_config": asdict(self.config), "history": history or [],
        })

    def load(self, path, map_location: Optional[str] = None) -> dict:
        from omt_branching.model.persistence import load_policy_into

        meta = load_policy_into(self.policy, path, map_location or self.config.device)
        self._baseline = float(meta.get("baseline", 0.0))
        return meta


def _categorical_entropy(probs: torch.Tensor) -> torch.Tensor:
    p = probs.clamp_min(1e-12)
    return -(p * torch.log(p)).sum()


def _bernoulli_entropy(p_true: torch.Tensor) -> torch.Tensor:
    p = p_true.clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


__all__ = [
    "DecideRLConfig",
    "DecideStep",
    "DecideEpisode",
    "SamplingDecider",
    "DecideRLTrainer",
]
