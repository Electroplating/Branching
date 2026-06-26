"""用 solver_sim 对比 GNN refocus、VSIDS、Oracle 三种策略。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.solver_sim import SimResult, problem_from_snapshot, simulate
from experiments.synthetic_omt import SyntheticConfig, generate_dataset
from omt_branching.service import BranchingPolicyService, ServiceConfig
from omt_branching.model.policy import BranchingPolicy, PolicyConfig

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SyntheticConfig(n_bool=30, n_numeric=12, n_clauses=60, n_atoms=20, seed=12345)
    snaps = generate_dataset(20, cfg)

    # 加载训练好的 baseline 模型（如果存在）
    policy_path = RESULTS_DIR / "baseline_policy.pt"
    if policy_path.exists():
        policy = BranchingPolicy(config=PolicyConfig(hidden=64, num_layers=3, use_auxiliary=True)).to(device)
        policy.load_state_dict(torch.load(policy_path, map_location=device))
        service = BranchingPolicyService(policy=policy)
        print("Loaded trained baseline model.")
    else:
        service = BranchingPolicyService()
        print("No trained model found, using random-initialized GNN.")

    results: dict[str, list[SimResult]] = {"gnn": [], "vsids": [], "oracle": []}
    for snap in snaps:
        prob = problem_from_snapshot(snap)
        results["gnn"].append(simulate(prob, "gnn", service, max_steps=80, seed=None))
        results["vsids"].append(simulate(prob, "vsids", max_steps=80, seed=None))
        results["oracle"].append(simulate(prob, "oracle", max_steps=80, seed=None))

    summary = {}
    for strat, res_list in results.items():
        avg_decisions = sum(r.decisions for r in res_list) / len(res_list)
        avg_conflicts = sum(r.conflicts for r in res_list) / len(res_list)
        avg_props = sum(r.propagations for r in res_list) / len(res_list)
        summary[strat] = {
            "avg_decisions": avg_decisions,
            "avg_conflicts": avg_conflicts,
            "avg_propagations": avg_props,
        }
        print(
            f"{strat:8s} | decisions={avg_decisions:7.1f} conflicts={avg_conflicts:7.1f} "
            f"props={avg_props:9.1f}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "sim_compare.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
