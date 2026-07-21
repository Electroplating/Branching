"""并行为数据集构建 VSIDS 轨迹标签缓存（``vsids_trace/<split>/<id>.json``）。

同一 ``.smt2`` + 相同 ``VSIDSTraceConfig(stride, max_examples)`` 下轨迹确定性；供
imitation 训练复用，避免每次重跑观察-only 求解。

运行::

    python -m examples.build_vsids_trace_cache
    python -m examples.build_vsids_trace_cache --dataset-dir examples/artifacts/dataset
    python -m examples.build_vsids_trace_cache --split train --workers 12 --max-ex 40 --force
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from omt_branching.solver.decide_omt import list_split_entries, smt2_to_instance
from omt_branching.solver.propagator_snapshot import prepare_propagator_formula
from omt_branching.solver.vsids_trace import VSIDSTraceConfig, collect_vsids_trajectory
from omt_branching.solver.vsids_trace_cache import (
    VSIDS_TRACE_SUBDIR,
    has_vsids_trace_result,
    save_vsids_trace_result,
)

ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")
DEFAULT_DATASET_DIR = os.path.join(ARTIFACTS, "dataset")
DEFAULT_WORKERS = max(1, min(12, (os.cpu_count() or 4)))


def _worker(task: tuple) -> dict:
    (
        dataset_dir,
        split,
        instance_id,
        smt2_relpath,
        stride,
        max_examples,
        force,
    ) = task
    if not force and has_vsids_trace_result(dataset_dir, instance_id, split=split):
        return {
            "instance_id": instance_id,
            "split": split,
            "skipped": True,
            "status": "cached",
        }

    smt2_path = Path(dataset_dir) / smt2_relpath
    if not smt2_path.is_file():
        return {
            "instance_id": instance_id,
            "split": split,
            "skipped": False,
            "status": "missing_smt2",
        }

    inst = smt2_to_instance(smt2_path, instance_id=instance_id)
    assertions, atoms = prepare_propagator_formula(list(inst.hard))
    cfg = VSIDSTraceConfig(stride=stride, max_examples=max_examples)
    records, ref_conflicts, info = collect_vsids_trajectory(
        assertions, atoms, cfg
    )
    # 空轨迹也落盘，供 imitation 识别「已采集」而非缺失
    save_vsids_trace_result(
        dataset_dir,
        instance_id,
        split=split,
        records=records,
        ref_conflicts=ref_conflicts,
        info=info,
        stride=stride,
        max_examples=max_examples,
    )
    return {
        "instance_id": instance_id,
        "split": split,
        "skipped": False,
        "status": "ok" if records else "empty",
        "n_records": len(records),
        "ref_conflicts": ref_conflicts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="并行构建 VSIDS 轨迹标签缓存")
    ap.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    ap.add_argument("--split", default=None, help="只处理某一划分（默认全部）")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument(
        "--max-ex",
        type=int,
        default=40,
        help="单实例样本上限（0=不限；写入缓存指纹）",
    )
    ap.add_argument("--stride", type=int, default=1, help="每 stride 次决策留 1 条")
    ap.add_argument("--force", action="store_true", help="覆盖已有缓存")
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    manifest_path = root / "manifest.json"
    splits: dict[str, list] = {}
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            splits = json.load(f).get("splits", {})
    keys = [args.split] if args.split else (list(splits.keys()) or ["test", "train"])

    tasks: list[tuple] = []
    for sp in keys:
        entries = splits.get(sp) or list_split_entries(args.dataset_dir, sp)
        for e in entries:
            tasks.append((
                args.dataset_dir,
                sp,
                e["instance_id"],
                e["smt2"],
                args.stride,
                args.max_ex,
                args.force,
            ))

    if not tasks:
        print("无可处理实例")
        return

    n_workers = max(1, min(args.workers, len(tasks)))
    print(
        f"vsids_trace 缓存: {len(tasks)} 实例, workers={n_workers}, "
        f"stride={args.stride}, max_ex={args.max_ex}, force={args.force}"
    )
    print(f"结果目录: {root / VSIDS_TRACE_SUBDIR}/")

    stats = {"cached": 0, "ok": 0, "empty": 0, "fail": 0}
    n_records = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="vsids_trace") as pbar:
            for fut in as_completed(futures):
                row = fut.result()
                st = row.get("status")
                if row.get("skipped"):
                    stats["cached"] += 1
                elif st == "ok":
                    stats["ok"] += 1
                    n_records += int(row.get("n_records") or 0)
                elif st == "empty":
                    stats["empty"] += 1
                else:
                    stats["fail"] += 1
                pbar.set_postfix(**stats)
                pbar.update(1)

    print(
        f"完成: cached={stats['cached']} ok={stats['ok']} "
        f"empty={stats['empty']} fail={stats['fail']} "
        f"records={n_records}"
    )


if __name__ == "__main__":
    main()
