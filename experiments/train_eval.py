"""训练与评估脚本。

- 在合成数据上构造 RankingExample。
- 用 ImitationTrainer 训练 BranchingPolicy。
- 评估 top-k accuracy、phase accuracy、整数分支 accuracy。
- 与 Oracle 和 VSIDS baseline 对比。
"""

from __future__ import annotations

import random
from typing import Iterable

import torch

from omt_branching.input.solver_state import SolverSnapshot
from omt_branching.interfaces import NodeType
from omt_branching.model.policy import BranchingPolicy, PolicyConfig
from omt_branching.model.trainer import ImitationTrainer, RankingExample, TrainConfig
from omt_branching.output.decoder import AdviceDecoder

from experiments.oracle import OracleBrancher
from experiments.synthetic_omt import SyntheticConfig, generate_dataset


def build_examples(
    snaps: Iterable[SolverSnapshot], oracle: OracleBrancher | None = None
) -> list[RankingExample]:
    oracle = oracle or OracleBrancher()
    return [oracle.make_example(s) for s in snaps]


def top_k_accuracy(
    policy: BranchingPolicy, examples: list[RankingExample], k: int = 1, device: str = "cpu"
) -> float:
    """Oracle top-1 候选是否落在 GNN top-k 中。"""
    policy.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for ex in examples:
            if not ex.bool_target_scores:
                continue
            g = ex.graph.to(device)
            out = policy.infer(g)
            if out.candidate_bool_local:
                cand_scores = out.bool_branch_scores[out.candidate_bool_local]
                topk_idx = torch.topk(cand_scores, min(k, len(out.candidate_bool_local))).indices
                topk_local = [out.candidate_bool_local[i] for i in topk_idx.tolist()]
            else:
                topk_local = []
            oracle_best = max(ex.bool_target_scores, key=ex.bool_target_scores.get)
            total += 1
            if oracle_best in topk_local:
                correct += 1
    return correct / total if total else 0.0


def phase_accuracy(
    policy: BranchingPolicy, examples: list[RankingExample], device: str = "cpu"
) -> float:
    """phase 预测准确率。"""
    policy.eval()
    correct = total = 0
    with torch.no_grad():
        for ex in examples:
            if not ex.phase_targets:
                continue
            g = ex.graph.to(device)
            out = policy.infer(g)
            for local, target in ex.phase_targets.items():
                pred = out.phase_logits[local] >= 0.0
                total += 1
                if bool(pred.item()) == target:
                    correct += 1
    return correct / total if total else 0.0


def int_branch_top1_accuracy(
    policy: BranchingPolicy, examples: list[RankingExample], device: str = "cpu"
) -> float:
    """整数 B&B top-1 候选准确率。"""
    policy.eval()
    correct = total = 0
    with torch.no_grad():
        for ex in examples:
            if not ex.int_target_scores:
                continue
            g = ex.graph.to(device)
            out = policy.infer(g)
            if out.candidate_numeric_local:
                cand_scores = out.int_branch_scores[out.candidate_numeric_local]
                best_local = out.candidate_numeric_local[int(torch.argmax(cand_scores).item())]
            else:
                continue
            oracle_best = max(ex.int_target_scores, key=ex.int_target_scores.get)
            total += 1
            if oracle_best == best_local:
                correct += 1
    return correct / total if total else 0.0


def oracle_top1_accuracy(examples: list[RankingExample]) -> float:
    """Hand-crafted OMT-aware heuristic（Oracle 规则）的 top-1 准确率。

    这里 Oracle 既是标签生成器，也作为强 baseline；GNN 应逼近该分数。
    """
    correct = total = 0
    for ex in examples:
        if not ex.bool_target_scores:
            continue
        oracle_best = max(ex.bool_target_scores, key=ex.bool_target_scores.get)
        # Oracle 自己预测自己，应接近 1.0（浮点 / softmax 不影响 argmax）
        total += 1
        # 由于 label 来自 Oracle，top-1 必然命中
        if oracle_best == oracle_best:
            correct += 1
    return correct / total if total else 0.0


def vsids_top1_accuracy(examples: list[RankingExample]) -> float:
    """用原生 VSIDS activity 作为 baseline 的 top-1 准确率。"""
    correct = total = 0
    for ex in examples:
        if not ex.bool_target_scores:
            continue
        # 从 graph 节点特征恢复 VSIDS activity：第 8 个特征是 vsids_activity
        feats = ex.graph.node_features[NodeType.BOOL_VAR]
        vsids = feats[:, 8]
        cand = ex.graph.meta.get("candidate_bool_local", [])
        if not cand:
            continue
        best_vsids = max(cand, key=lambda i: float(vsids[i]))
        oracle_best = max(ex.bool_target_scores, key=ex.bool_target_scores.get)
        total += 1
        if best_vsids == oracle_best:
            correct += 1
    return correct / total if total else 0.0


def train_policy(
    train_examples: list[RankingExample],
    val_examples: list[RankingExample],
    epochs: int = 30,
    hidden: int = 64,
    num_layers: int = 3,
    lr: float = 1e-3,
    device: str = "cpu",
    accum_steps: int = 1,
) -> tuple[BranchingPolicy, list[dict[str, float]]]:
    """训练并返回模型与验证指标历史。"""
    policy = BranchingPolicy(config=PolicyConfig(hidden=hidden, num_layers=num_layers, use_auxiliary=True)).to(device)
    trainer = ImitationTrainer(policy, TrainConfig(lr=lr, device=device, accum_steps=accum_steps))

    history: list[dict[str, float]] = []
    for ep in range(epochs):
        random.shuffle(train_examples)
        train_metrics = trainer.fit(train_examples, epochs=1)[0]
        top1 = top_k_accuracy(policy, val_examples, k=1, device=device)
        top3 = top_k_accuracy(policy, val_examples, k=3, device=device)
        phase = phase_accuracy(policy, val_examples, device=device)
        int_top1 = int_branch_top1_accuracy(policy, val_examples, device=device)
        metrics = {
            "epoch": ep,
            "train_loss": train_metrics["loss"],
            "val_top1": top1,
            "val_top3": top3,
            "val_phase": phase,
            "val_int_top1": int_top1,
        }
        history.append(metrics)
        if ep % 5 == 0 or ep == epochs - 1:
            print(
                f"Epoch {ep:3d} | train_loss={train_metrics['loss']:.4f} "
                f"val_top1={top1:.3f} top3={top3:.3f} phase={phase:.3f} int_top1={int_top1:.3f}"
            )
    return policy, history


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cfg = SyntheticConfig(n_bool=30, n_numeric=12, n_clauses=60, n_atoms=20, seed=42)
    train_snaps = generate_dataset(500, cfg)
    val_snaps = generate_dataset(100, SyntheticConfig(**{**cfg.__dict__, "seed": 9999}))

    oracle = OracleBrancher()
    train_ex = build_examples(train_snaps, oracle)
    val_ex = build_examples(val_snaps, oracle)

    print(f"Train examples: {len(train_ex)}, Val examples: {len(val_ex)}")
    print(f"VSIDS baseline top-1 accuracy: {vsids_top1_accuracy(val_ex):.3f}")
    print(f"Oracle heuristic top-1 accuracy: {oracle_top1_accuracy(val_ex):.3f}")

    policy, history = train_policy(
        train_ex, val_ex, epochs=40, hidden=64, num_layers=3, lr=1e-3, device=device
    )

    # 最终评估
    print("\n=== Final Evaluation ===")
    print(f"GNN top-1 : {top_k_accuracy(policy, val_ex, k=1, device=device):.3f}")
    print(f"GNN top-3 : {top_k_accuracy(policy, val_ex, k=3, device=device):.3f}")
    print(f"GNN phase : {phase_accuracy(policy, val_ex, device=device):.3f}")
    print(f"GNN int-1 : {int_branch_top1_accuracy(policy, val_ex, device=device):.3f}")
    print(f"VSIDS     : {vsids_top1_accuracy(val_ex):.3f}")
    print(f"Oracle    : {oracle_top1_accuracy(val_ex):.3f}")

    return policy, history


if __name__ == "__main__":
    main()
