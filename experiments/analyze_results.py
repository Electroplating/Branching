"""读取 experiments/results/ 下的 JSON 结果并生成简要报告。"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    summary_path = RESULTS_DIR / "summary.json"
    if not summary_path.exists():
        print(f"No summary found at {summary_path}")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    print("\n" + "=" * 70)
    print("Experiment Summary")
    print("=" * 70)
    print(
        f"{'Experiment':<25} {'top1':>6} {'top3':>6} {'phase':>6} "
        f"{'int1':>6} {'vsids':>6} {'oracle':>6}"
    )
    for name, metrics in summary.items():
        print(
            f"{name:<25} {metrics['gnn_top1']:>6.3f} {metrics['gnn_top3']:>6.3f} "
            f"{metrics['gnn_phase']:>6.3f} {metrics['gnn_int_top1']:>6.3f} "
            f"{metrics['vsids_top1']:>6.3f} {metrics['oracle_top1']:>6.3f}"
        )

    # 打印每个实验的详细训练曲线
    for exp_file in sorted(RESULTS_DIR.glob("*.json")):
        if exp_file.name in ("summary.json", "sim_compare.json"):
            continue
        with open(exp_file) as f:
            data = json.load(f)
        print(f"\n--- {data['name']} ---")
        print(f"Total time: {data['total_time']:.1f}s")
        last = data["history"][-1] if data["history"] else {}
        print(f"Final train loss: {last.get('train_loss', -1):.4f}")


if __name__ == "__main__":
    main()
