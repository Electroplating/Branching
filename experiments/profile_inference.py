"""测量 GNN 推理开销随图规模的变化。

输出：
- 不同 n_bool/numeric/clause 规模下的 forward 耗时（ms）
- 节点数 / 边数
- 用于评估部署时的 overhead 与规模门限设定。
"""

from __future__ import annotations

import time

import torch

from experiments.synthetic_omt import SyntheticConfig, generate_snapshot
from omt_branching.input.graph_builder import GraphBuilder
from omt_branching.model.policy import BranchingPolicy, PolicyConfig
from omt_branching.service import BranchingPolicyService


def profile(scale: int, device: str = "cpu") -> dict:
    cfg = SyntheticConfig(
        n_bool=scale,
        n_numeric=max(5, scale // 3),
        n_clauses=scale * 2,
        n_atoms=max(5, scale // 2),
        seed=42,
    )
    snap = generate_snapshot(cfg)
    builder = GraphBuilder()
    g = builder.build(snap)

    policy = BranchingPolicy(config=PolicyConfig(hidden=64, num_layers=3, use_auxiliary=True)).to(device)
    policy.eval()
    g = g.to(device)

    # warmup
    with torch.no_grad():
        for _ in range(3):
            _ = policy.infer(g)

    n_repeats = 20
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            torch.cuda.synchronize() if device == "cuda" else None
            t0 = time.perf_counter()
            _ = policy.infer(g)
            torch.cuda.synchronize() if device == "cuda" else None
            times.append((time.perf_counter() - t0) * 1000.0)

    total_nodes = sum(g.num_nodes(nt) for nt in g.node_types())
    total_edges = sum(g.num_edges(et) for et in g.edge_types())
    return {
        "scale": scale,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"{'scale':>8} {'nodes':>8} {'edges':>9} {'mean_ms':>10} {'min_ms':>10} {'max_ms':>10}")
    for scale in [10, 20, 50, 100, 200, 500]:
        r = profile(scale, device)
        print(
            f"{r['scale']:>8} {r['total_nodes']:>8} {r['total_edges']:>9} "
            f"{r['mean_ms']:>10.3f} {r['min_ms']:>10.3f} {r['max_ms']:>10.3f}"
        )


if __name__ == "__main__":
    main()
