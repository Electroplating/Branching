"""Phase 2 端到端：look-ahead imitation 冷启动 -> decide 路径 RL 微调 -> 三臂对比。

在带布尔结构的有界整数 OMT（``generate_bool_lia_dataset``，含 ``Or`` 析取 -> 真正的
内部 case-split）上：

1. **imitation 冷启动**：SAT look-ahead 教师（试探赋值 + 单元传播计数）给布尔原子打分，
   用 :class:`ImitationTrainer` 训练 bool head + phase head（``L_branch + λ·L_phase``）。
2. **Solver-in-the-Loop RL**：:class:`DecideRLTrainer` 在 z3 内部布尔决策回路里用 REINFORCE
   微调，奖励 = ``-log(1+rlimit)``（整体求解开销）。
3. **三臂对比**：native(z3 Optimize) / VSIDS-decide / learned-decide（训练后），报告
   rlimit/conflicts/decisions 与 ``match``（== native 的正确性）。

**成功标准（spec §6 Phase 2）**：learned-decide 在 rlimit 上优于 VSIDS-decide，且 match=1。

快速冒烟::

    python -m examples.decide_learn --train 8 --test 6 --min-vars 4 --max-vars 4 --iters 1 --epochs 5
"""

from __future__ import annotations

import argparse
import os

import torch

from omt_branching.model.persistence import load_policy, save_history, save_policy
from omt_branching.model.policy import BranchingPolicy
from omt_branching.model.trainer import ImitationTrainer, TrainConfig
from omt_branching.service import BranchingPolicyService
from omt_branching.solver import (
    DecideRLConfig, DecideRLTrainer, PolicyDecider, Z3Backend,
    build_lookahead_examples, decide_bool_hit, generate_bool_lia_dataset,
    solve_native, solve_omt_with_decider,
)

ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")
GNN_CKPT = os.path.join(ARTIFACTS, "gnn_decide.pt")
RL_CKPT = os.path.join(ARTIFACTS, "rl_decide_policy.pt")
HISTORY_JSON = os.path.join(ARTIFACTS, "rl_decide_history.json")


def _native_rlimit(hard, obj, sense) -> int:
    b = Z3Backend()
    b.optimize(b.conjoin(*hard), obj, sense)
    return b.rlimit_count


def branch_accuracy(policy, instances) -> tuple[float, int]:
    """bool-head top-1 与 look-ahead 专家一致的比例，及有效实例数。"""
    hit = total = 0
    for inst in instances:
        r = decide_bool_hit(policy, list(inst.hard))
        if r is None:
            continue
        total += 1
        hit += 1 if r else 0
    return (hit / total if total else 0.0), total


def three_arm(policy, instances, refocus: int) -> dict:
    """native / VSIDS-decide / learned-decide 三臂对比（learned 用给定 policy）。"""
    svc = BranchingPolicyService(policy=policy)
    agg = {"native": {"rlimit": 0.0},
           "vsids": {"rlimit": 0.0, "conflicts": 0.0, "match": 0.0},
           "learned": {"rlimit": 0.0, "conflicts": 0.0, "decisions": 0.0, "match": 0.0}}
    for inst in instances:
        hard, obj, sense = inst.as_tuple()
        native = solve_native(hard, obj, sense)
        agg["native"]["rlimit"] += _native_rlimit(hard, obj, sense)

        v = solve_omt_with_decider(hard, obj, sense, decider_factory=None)
        agg["vsids"]["rlimit"] += v["rlimit"]
        agg["vsids"]["conflicts"] += v["conflicts"]
        agg["vsids"]["match"] += 1.0 if v["value"] == native else 0.0

        ln = solve_omt_with_decider(
            hard, obj, sense,
            decider_factory=lambda a: PolicyDecider(svc, a, refocus))
        agg["learned"]["rlimit"] += ln["rlimit"]
        agg["learned"]["conflicts"] += ln["conflicts"]
        agg["learned"]["decisions"] += ln["decisions"]
        agg["learned"]["match"] += 1.0 if ln["value"] == native else 0.0
    n = max(1, len(instances))
    for arm in agg.values():
        for m in arm:
            arm[m] /= n
    return agg


def _print_arms(tag: str, agg: dict) -> None:
    print(f"[{tag}] rlimit/conflicts 越小越好，match=1 为正确：")
    print(f"  native(z3 Optimize): rlimit={agg['native']['rlimit']:.0f}")
    print(f"  VSIDS-decide       : rlimit={agg['vsids']['rlimit']:.0f} "
          f"conflicts={agg['vsids']['conflicts']:.1f} match={agg['vsids']['match']:.2f}")
    print(f"  learned-decide     : rlimit={agg['learned']['rlimit']:.0f} "
          f"conflicts={agg['learned']['conflicts']:.1f} "
          f"decisions={agg['learned']['decisions']:.1f} match={agg['learned']['match']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2：look-ahead imitation + decide 路径 RL")
    ap.add_argument("--train", type=int, default=60)
    ap.add_argument("--test", type=int, default=20)
    ap.add_argument("--min-vars", type=int, default=4)
    ap.add_argument("--max-vars", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20, help="imitation 轮数")
    ap.add_argument("--iters", type=int, default=2, help="RL 训练轮数")
    ap.add_argument("--refocus", type=int, default=30)
    ap.add_argument("--rl-log", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    os.makedirs(ARTIFACTS, exist_ok=True)

    # ---------------- 1) 生成布尔结构整数 OMT 实例 ----------------
    print("=== 1) 生成 OMT(bool-LIA) 实例（含 Or 析取 -> 内部 case-split） ===")
    train_set = generate_bool_lia_dataset(args.train, seed=1,
                                          min_vars=args.min_vars, max_vars=args.max_vars)
    test_set = generate_bool_lia_dataset(args.test, seed=99,
                                         min_vars=args.min_vars, max_vars=args.max_vars)
    print(f"训练集 {len(train_set)} 个，测试集 {len(test_set)} 个 "
          f"(vars {args.min_vars}..{args.max_vars})")

    # ---------------- 2) look-ahead imitation 冷启动 ----------------
    print("\n=== 2) look-ahead imitation 冷启动 (bool head + phase head) ===")
    policy = BranchingPolicy()
    acc_before, n_valid = branch_accuracy(policy, test_set)
    print(f"训练前 bool 分支准确率(vs look-ahead 专家): {acc_before:.2f} (有效 {n_valid} 例)")

    examples = build_lookahead_examples([list(i.hard) for i in train_set])
    print(f"look-ahead imitation 样本数: {len(examples)}")
    trainer = ImitationTrainer(policy, TrainConfig(lr=5e-3))
    hist_imit = trainer.fit(examples, epochs=args.epochs)
    print(f"imitation loss: 首轮 {hist_imit[0]['loss']:.4f} -> 末轮 {hist_imit[-1]['loss']:.4f}")
    acc_after, _ = branch_accuracy(policy, test_set)
    print(f"训练后 bool 分支准确率: {acc_after:.2f}")
    save_policy(policy, GNN_CKPT, meta={"stage": "imitation", "theory": "bool-LIA"})

    print("\n--- imitation 后三臂对比（测试集） ---")
    _print_arms("imitation", three_arm(policy, test_set, args.refocus))

    # ---------------- 3) Solver-in-the-Loop RL 微调 ----------------
    print("\n=== 3) Solver-in-the-Loop RL 微调 (REINFORCE, -log(1+rlimit)) ===")
    rl_cfg = DecideRLConfig(lr=1e-3, gamma=0.98, entropy_coef=5e-3,
                            refocus_every=args.refocus, rlimit_penalty_coef=1.0)
    rl_trainer = DecideRLTrainer(policy, rl_cfg)
    instances = [inst.as_tuple() for inst in train_set]
    hist_rl = rl_trainer.train(instances, iterations=args.iters, log=args.rl_log)
    if hist_rl:
        print(f"RL return: 首条 {hist_rl[0]['return']:.4f} -> 末条 {hist_rl[-1]['return']:.4f} "
              f"(记录 {len(hist_rl)} 条)")
    rl_trainer.save(RL_CKPT, history=hist_rl)
    save_history(hist_rl, HISTORY_JSON)

    reloaded, meta = load_policy(RL_CKPT)
    print(f"重载 checkpoint：kind={meta.get('kind')} baseline={meta.get('baseline'):.4f} "
          f"history_len={len(meta.get('history', []))}")

    # ---------------- 4) RL 后三臂对比 ----------------
    print("\n=== 4) RL 微调后三臂对比（测试集） ===")
    agg = three_arm(reloaded, test_set, args.refocus)
    _print_arms("RL", agg)

    rl_l, rl_v = agg["learned"]["rlimit"], agg["vsids"]["rlimit"]
    verdict = "优于" if rl_l < rl_v else ("持平" if abs(rl_l - rl_v) < 1e-6 else "不及")
    print(f"\nPhase 2 结论：learned-decide rlimit={rl_l:.0f} 相对 VSIDS-decide rlimit={rl_v:.0f} "
          f"{verdict}；match={agg['learned']['match']:.2f}（应为 1.00，正确性 == native）。")


if __name__ == "__main__":
    main()
