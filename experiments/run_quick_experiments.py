"""快速实验脚本：更小规模，用于快速验证与演示。"""

from __future__ import annotations

from experiments.run_experiments import run_experiment
from experiments.synthetic_omt import SyntheticConfig

import torch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    base_cfg = SyntheticConfig(n_bool=30, n_numeric=12, n_clauses=60, n_atoms=20, seed=42)
    val_cfg = SyntheticConfig(**{**base_cfg.__dict__, "seed": 9999})

    # 快速基准：小 hidden、少 layer、少 epoch
    r = run_experiment(
        "quick_baseline",
        base_cfg, val_cfg,
        n_train=80, n_val=20, epochs=15,
        hidden=32, num_layers=2, lr=1e-3, device=device,
    )
    print("\nQuick baseline final:", {k: f"{v:.3f}" for k, v in r["final"].items()})


if __name__ == "__main__":
    main()
