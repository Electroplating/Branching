"""在线 REINFORCE 微调演示。

在合成 snapshots 上模拟 solver-in-the-loop：
- 策略选择 top-1 bool 候选。
- 若与 Oracle 推荐一致，获得 step reward +1；否则 -1。
- 在轨迹末尾给一个 terminal reward（与一致率相关）。

展示如何从 imitation checkpoint 继续提升 top-1 accuracy。
"""

from __future__ import annotations

import random

import torch

from experiments.oracle import OracleBrancher
from experiments.synthetic_omt import SyntheticConfig, generate_dataset
from experiments.train_eval import build_examples, top_k_accuracy
from omt_branching.interfaces import NodeType
from omt_branching.model.finetune import FinetuneConfig, SolverInLoopFinetuner, Trajectory, TrajectoryStep
from omt_branching.model.policy import BranchingPolicy, PolicyConfig
from omt_branching.service import BranchingPolicyService


def collect_trajectory(policy, snap, oracle, device="cpu"):
    """对一个快照收集多步 trajectory（模拟 solver 决策序列）。"""
    from experiments.solver_sim import _build_snapshot, _fresh_state, problem_from_snapshot

    prob = problem_from_snapshot(snap)
    state = _fresh_state(prob)
    service = BranchingPolicyService(policy=policy)
    steps = []
    max_steps = 10
    for _ in range(max_steps):
        if len(state.assignment) >= len(prob.bool_vars):
            break
        cur_snap = _build_snapshot(prob, state, len(steps))
        advice = service.advise(cur_snap)
        if not advice.ranked_candidates:
            break
        chosen_id = advice.ranked_candidates[0]
        chosen_local = service.builder.build(cur_snap).id_maps[NodeType.BOOL_VAR][chosen_id]
        oracle_scores = oracle.bool_branch_scores(cur_snap)
        oracle_best = max(oracle_scores, key=oracle_scores.get) if oracle_scores else None
        reward = 1.0 if chosen_id == oracle_best else -1.0
        steps.append(TrajectoryStep(graph=service.builder.build(cur_snap), chosen_bool_local=chosen_local, reward=reward))
        # 模拟赋值
        from experiments.solver_sim import _apply_bool_decision
        _apply_bool_decision(prob, state, chosen_id, advice.phase_suggestions.get(chosen_id, True))
    consistency = sum(1 for s in steps if s.reward > 0) / max(1, len(steps))
    return Trajectory(steps=steps, terminal_reward=consistency * 5.0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SyntheticConfig(n_bool=30, n_numeric=12, n_clauses=60, n_atoms=20, seed=42)
    oracle = OracleBrancher()
    val_ex = build_examples(generate_dataset(20, SyntheticConfig(**{**cfg.__dict__, "seed": 9999})), oracle)

    policy = BranchingPolicy(config=PolicyConfig(hidden=64, num_layers=3, use_auxiliary=True)).to(device)
    finetuner = SolverInLoopFinetuner(policy, FinetuneConfig(lr=3e-4, device=device))

    print("Before REINFORCE top-1:", top_k_accuracy(policy, val_ex, k=1, device=device))

    train_snaps = generate_dataset(50, cfg)
    for ep in range(10):
        random.shuffle(train_snaps)
        ep_reward = 0.0
        for snap in train_snaps:
            traj = collect_trajectory(policy, snap, oracle, device=device)
            stats = finetuner.reinforce_update(traj)
            ep_reward += sum(s.reward for s in traj.steps)
        acc = top_k_accuracy(policy, val_ex, k=1, device=device)
        print(f"Epoch {ep} | avg step reward {ep_reward/len(train_snaps):.2f} | val top-1 {acc:.3f}")


if __name__ == "__main__":
    main()
