"""测试 torch.compile 对 BranchingPolicy 推理的加速效果。"""

from __future__ import annotations

import time

import torch

from experiments.synthetic_omt import SyntheticConfig, generate_snapshot
from omt_branching.input.graph_builder import GraphBuilder
from omt_branching.model.policy import BranchingPolicy, PolicyConfig


def measure(policy, g, device, repeats=50):
    policy.eval()
    g = g.to(device)
    with torch.no_grad():
        # warmup
        for _ in range(5):
            _ = policy.infer(g)
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = policy.infer(g)
        torch.cuda.synchronize() if device == "cuda" else None
        return (time.perf_counter() - t0) / repeats * 1000.0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SyntheticConfig(n_bool=100, n_numeric=30, n_clauses=200, n_atoms=50, seed=42)
    snap = generate_snapshot(cfg)
    g = GraphBuilder().build(snap)

    policy = BranchingPolicy(config=PolicyConfig(hidden=64, num_layers=3, use_auxiliary=True)).to(device)
    t_eager = measure(policy, g, device)

    try:
        compiled = torch.compile(policy)
        t_compiled = measure(compiled, g, device)
        print(f"Eager:   {t_eager:.3f} ms")
        print(f"Compiled:{t_compiled:.3f} ms")
        print(f"Speedup: {t_eager / t_compiled:.2f}x")
    except Exception as e:
        print(f"torch.compile not available or failed: {e}")
        print(f"Eager: {t_eager:.3f} ms")


if __name__ == "__main__":
    main()
