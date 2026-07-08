"""困难 SMT(LIA) 学习分支：紧随机整数线性算术实例。两臂均附 propagator（关预处理→纯
CDCL），比 VSIDS-decide vs (imitation+RL) learned-decide 的 conflicts。原子=线性算术
比较（理论原子，非布尔变量）。多 seed mean±std。
"""
from __future__ import annotations

import argparse
import statistics

import torch

from omt_branching.model.policy import BranchingPolicy
from omt_branching.model.trainer import ImitationTrainer, TrainConfig
from omt_branching.service import BranchingPolicyService
from omt_branching.solver import generate_hard_smt_lia, solve_sat_with_decider
from omt_branching.solver.policy_decider import PolicyDecider
from omt_branching.solver.rl_decide import DecideRLConfig, DecideRLTrainer
from omt_branching.solver.training_data import build_lookahead_examples_sat


def _bench(problems, decider_factory):
    confs = []
    for atoms, clauses in problems:
        r = solve_sat_with_decider(clauses, atoms, decider_factory=decider_factory)
        confs.append(r["conflicts"])
    return confs


def main() -> None:
    ap = argparse.ArgumentParser(description="困难 SMT(LIA) 学习分支：VSIDS vs learned")
    ap.add_argument("--n-vars", type=int, default=8)
    ap.add_argument("--n-disj", type=int, default=30)
    ap.add_argument("--test", type=int, default=12)
    ap.add_argument("--train", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--rl-iters", type=int, default=0)
    ap.add_argument("--refocus", type=int, default=100)
    args = ap.parse_args()
    torch.manual_seed(0)

    test = [generate_hard_smt_lia(args.n_vars, args.n_disj, 3, 6, 4, 1000 + s)
            for s in range(args.test)]

    policy = BranchingPolicy()
    if args.train > 0:
        tr = [generate_hard_smt_lia(args.n_vars, args.n_disj, 3, 6, 4, s) for s in range(args.train)]
        exs = [e for e in build_lookahead_examples_sat(tr) if e.bool_target_scores]
        h = ImitationTrainer(policy, TrainConfig(lr=5e-3)).fit(exs, epochs=args.epochs)
        print(f"imitation: {len(exs)} 样本, branch {h[0].get('branch', 0):.3f}->{h[-1].get('branch', 0):.3f}")
    if args.rl_iters > 0:
        rl = [generate_hard_smt_lia(args.n_vars, args.n_disj, 3, 6, 4, s) for s in range(max(args.train, 20))]
        rlt = DecideRLTrainer(policy, DecideRLConfig(refocus_every=args.refocus))
        hh = rlt.train_sat(rl, iterations=args.rl_iters, log=False)
        if hh:
            print(f"RL: {len(hh)} 步, 末条 conflicts={hh[-1]['conflicts']}, "
                  f"defer_logit={float(rlt.defer_logit):.3f}")
    svc = BranchingPolicyService(policy=policy)

    def learned_factory(assertions):
        return PolicyDecider(svc, assertions, args.refocus)

    v = _bench(test, None)
    ln = _bench(test, learned_factory)
    print(f"[SMT-LIA] VSIDS conflicts={statistics.fmean(v):.0f}±{statistics.pstdev(v):.0f} | "
          f"learned={statistics.fmean(ln):.0f}±{statistics.pstdev(ln):.0f} | "
          f"胜={'是' if statistics.fmean(ln) < statistics.fmean(v) else '否'}")
    print("两臂均附 propagator（关预处理→纯 CDCL）；分支于线性算术原子；conflicts 越少越好。")


if __name__ == "__main__":
    main()
