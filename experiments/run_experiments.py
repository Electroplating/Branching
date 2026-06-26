"""主实验脚本：合成数据 + 训练 + 多 baseline 对比 + 消融。

输出：
- 终端打印指标表格
- experiments/results/ 下的 JSON 日志与模型 checkpoint
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.oracle import OracleBrancher
from experiments.synthetic_omt import SyntheticConfig, generate_dataset
from experiments.train_eval import (
    build_examples,
    int_branch_top1_accuracy,
    oracle_top1_accuracy,
    phase_accuracy,
    top_k_accuracy,
    train_policy,
    vsids_top1_accuracy,
)


RESULTS_DIR = Path(__file__).parent / "results"


def run_experiment(
    name: str,
    train_cfg: SyntheticConfig,
    val_cfg: SyntheticConfig,
    n_train: int = 200,
    n_val: int = 50,
    epochs: int = 30,
    hidden: int = 64,
    num_layers: int = 3,
    lr: float = 1e-3,
    accum_steps: int = 1,
    device: str = "cpu",
) -> dict:
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    t0 = time.time()
    train_snaps = generate_dataset(n_train, train_cfg)
    val_snaps = generate_dataset(n_val, val_cfg)
    data_time = time.time() - t0

    oracle = OracleBrancher()
    train_ex = build_examples(train_snaps, oracle)
    val_ex = build_examples(val_snaps, oracle)

    print(f"Data generation: {data_time:.1f}s | Train: {len(train_ex)} | Val: {len(val_ex)}")

    vsids = vsids_top1_accuracy(val_ex)
    oracle = oracle_top1_accuracy(val_ex)
    print(f"VSIDS baseline top-1: {vsids:.3f}")
    print(f"Oracle heuristic top-1: {oracle:.3f}")

    policy, history = train_policy(
        train_ex, val_ex, epochs=epochs, hidden=hidden,
        num_layers=num_layers, lr=lr, accum_steps=accum_steps, device=device,
    )

    final = {
        "gnn_top1": top_k_accuracy(policy, val_ex, k=1, device=device),
        "gnn_top3": top_k_accuracy(policy, val_ex, k=3, device=device),
        "gnn_phase": phase_accuracy(policy, val_ex, device=device),
        "gnn_int_top1": int_branch_top1_accuracy(policy, val_ex, device=device),
        "vsids_top1": vsids,
        "oracle_top1": oracle,
    }
    print("Final:", {k: f"{v:.3f}" for k, v in final.items()})

    result = {
        "name": name,
        "config": {
            "train": asdict(train_cfg),
            "val": asdict(val_cfg),
            "n_train": n_train,
            "n_val": n_val,
            "epochs": epochs,
            "hidden": hidden,
            "num_layers": num_layers,
            "lr": lr,
        },
        "history": history,
        "final": final,
        "data_time": data_time,
        "total_time": time.time() - t0,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{name}.json", "w") as f:
        json.dump(result, f, indent=2)

    torch.save(policy.state_dict(), RESULTS_DIR / f"{name}_policy.pt")
    return result


def main():
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    base_cfg = SyntheticConfig(n_bool=30, n_numeric=12, n_clauses=60, n_atoms=20, seed=42)
    val_cfg = SyntheticConfig(**{**base_cfg.__dict__, "seed": 9999})

    experiments = [
        ("baseline", base_cfg, val_cfg, 64, 3),
        ("deep_gnn", base_cfg, val_cfg, 64, 4),
        ("wide_gnn", base_cfg, val_cfg, 128, 3),
    ]
    big_cfg = SyntheticConfig(n_bool=60, n_numeric=25, n_clauses=120, n_atoms=40, seed=12345)
    experiments.append(("generalization_large", base_cfg, big_cfg, 64, 3))

    results = []
    for name, train_cfg, test_cfg, hidden, layers in experiments:
        if only and name != only:
            continue
        results.append(run_experiment(
            name,
            train_cfg, test_cfg,
            n_train=120, n_val=30, epochs=20,
            hidden=hidden, num_layers=layers, lr=1e-3, accum_steps=4, device=device,
        ))

    if len(results) > 1:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"{'Experiment':<25} {'top1':>6} {'top3':>6} {'phase':>6} {'int1':>6} {'vsids':>6} {'oracle':>6}")
        for r in results:
            f = r["final"]
            print(
                f"{r['name']:<25} {f['gnn_top1']:>6.3f} {f['gnn_top3']:>6.3f} "
                f"{f['gnn_phase']:>6.3f} {f['gnn_int_top1']:>6.3f} {f['vsids_top1']:>6.3f} "
                f"{f['oracle_top1']:>6.3f}"
            )

        with open(RESULTS_DIR / "summary.json", "w") as f:
            json.dump({r["name"]: r["final"] for r in results}, f, indent=2)


if __name__ == "__main__":
    main()
