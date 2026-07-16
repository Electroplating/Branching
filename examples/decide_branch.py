"""三臂对比：z3 二进制参考 / VSIDS-decide / learned-decide。

以数据集中缓存的 ``solve_binary`` 结果为参考最优值，测量 VSIDS/learned 相对参考的
正确性（match）与 rlimit/conflicts/decisions。为 Phase 2（look-ahead imitation + RL）铺路。

数据集：若 ``--dataset-dir`` 已有 ``manifest.json``，用 ``smt2_to_instance`` /
``load_dataset`` 读取，不再重新生成；否则按参数生成并落盘。

binary 参考解须事先由 ``python -m examples.solve_dataset_binary`` 写入
``binary/<split>/<id>.json``；本脚本测试臂直接读缓存。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import torch

from omt_branching.model.device import gnn_device
from omt_branching.model.inference import InferenceConfig
from omt_branching.model.persistence import save_history
from omt_branching.model.policy import BranchingPolicy
from omt_branching.service import BranchingPolicyService, ServiceConfig
from omt_branching.solver.rl_decide import (
    DEFAULT_RL_COLLECT_WORKERS,
    DecideRLConfig,
    DecideRLTrainer,
    effective_rl_workers,
)
from omt_branching.solver import (
    generate_bool_lia_dataset,
    load_dataset,
    list_split_entries,
    manifest_mismatches,
    rebuild_manifest,
    save_dataset,
    solve_omt_with_decider,
)
from omt_branching.solver.binary_results import (
    load_binary_result,
    missing_binary_ids,
)
from omt_branching.solver.instance_gen import OMTInstance
from omt_branching.solver.policy_decider import PolicyDecider

from tqdm import tqdm

ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")
DEFAULT_DATASET_DIR = os.path.join(ARTIFACTS, "decide_branch_dataset")
DEFAULT_CKPT_DIR = os.path.join(ARTIFACTS, "rl_checkpoints")
DEFAULT_TEST_WORKERS = 30


def _json_value(v):
    """把 Fraction / 其它非标量转为 JSON 可序列化形式。"""
    if v is None:
        return None
    if isinstance(v, Fraction):
        return str(v)
    return v


def _stats_for_json(stats: dict) -> dict:
    return {k: _json_value(v) for k, v in stats.items()}


def dataset_exists(dataset_dir: str) -> bool:
    root = Path(dataset_dir)
    if (root / "manifest.json").is_file():
        return True
    return any((root / sp).glob("*.smt2") for sp in ("test", "train"))


def _sync_manifest_if_needed(dataset_dir: str) -> None:
    """若 manifest 与磁盘 .smt2 不一致，按磁盘重建。"""
    issues = manifest_mismatches(dataset_dir)
    if not issues:
        return
    print("检测到 manifest 与磁盘 .smt2 不一致，按磁盘重建 manifest：")
    for msg in issues:
        print(f"  - {msg}")
    rebuild_manifest(dataset_dir, preserve_meta=True)
    print(f"已重建 -> {Path(dataset_dir) / 'manifest.json'}")


def _smt2_abs_paths(dataset_dir: str, entries: list[dict]) -> list[str]:
    root = Path(dataset_dir)
    return [str(root / e["smt2"]) for e in entries]


def _require_binary_cache(dataset_dir: str, split: str, entries: list[dict]) -> None:
    missing = missing_binary_ids(dataset_dir, entries, split=split)
    if missing:
        preview = ", ".join(missing[:5])
        more = f" 等共 {len(missing)} 个" if len(missing) > 5 else ""
        raise SystemExit(
            f"划分 {split} 缺少 binary 缓存（如 {preview}{more}）。\n"
            f"请先运行: python -m examples.solve_dataset_binary "
            f"--dataset-dir {dataset_dir}"
        )


def _eval_test_worker(task: tuple) -> dict:
    """ProcessPool worker：从 .smt2 加载实例，binary 用缓存，跑 VSIDS/learned。"""
    (
        smt2_path,
        instance_id,
        binary_result,
        policy_state,
        device,
        refocus,
    ) = task
    from omt_branching.solver.decide_omt import smt2_to_instance

    inst = smt2_to_instance(smt2_path, instance_id=instance_id)
    hard, obj, sense = inst.as_tuple()
    ref = binary_result
    ref_val = ref.get("value")
    v = solve_omt_with_decider(hard, obj, sense, decider_factory=None)
    policy = BranchingPolicy()
    policy.load_state_dict(policy_state)
    policy.to(device)
    policy.eval()
    svc = BranchingPolicyService(
        policy=policy,
        config=ServiceConfig(inference=InferenceConfig(device=device)),
    )
    ln = solve_omt_with_decider(
        hard,
        obj,
        sense,
        decider_factory=lambda a: PolicyDecider(svc, a, refocus),
    )
    return {
        "instance_id": inst.instance_id,
        "ref_val": ref_val,
        "binary": ref,
        "vsids": v,
        "learned": ln,
    }


def _policy_state_cpu(policy: BranchingPolicy) -> dict:
    return {k: v.detach().cpu() for k, v in policy.state_dict().items()}


def _run_test_parallel(
    entries: list[dict],
    dataset_dir: str,
    policy: BranchingPolicy,
    device: str,
    refocus: int,
    workers: int,
) -> list[dict]:
    """并发跑测试集（进程池；binary 读缓存；每 worker 从 smt2 加载）。"""
    policy_state = _policy_state_cpu(policy)
    n_workers = max(1, min(workers, len(entries)))
    worker_device = device if n_workers == 1 else "cpu"
    root = Path(dataset_dir)
    tasks = []
    for e in entries:
        iid = e["instance_id"]
        cached = load_binary_result(dataset_dir, iid, split="test")
        if cached is None:
            raise RuntimeError(f"缺少 binary 缓存: {iid}")
        tasks.append((
            str(root / e["smt2"]),
            iid,
            cached,
            policy_state,
            worker_device,
            refocus,
        ))
    by_id: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_eval_test_worker, t): t[1] for t in tasks}
        with tqdm(total=len(entries), desc="test") as pbar:
            for fut in as_completed(futures):
                row = fut.result()
                by_id[row["instance_id"]] = row
                pbar.update(1)
    return [by_id[e["instance_id"]] for e in entries]


def main() -> None:
    ap = argparse.ArgumentParser(description="UserPropagator 学习分支三臂对比（binary 参考）")
    ap.add_argument("--test", type=int, default=20)
    ap.add_argument("--min-vars", type=int, default=4)
    ap.add_argument("--max-vars", type=int, default=5)
    ap.add_argument("--refocus", type=int, default=50)
    ap.add_argument(
        "--train", type=int, default=0, help="look-ahead imitation 训练集规模(0=不训练)"
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--rl-iters", type=int, default=0, help="RL 微调轮数(0=不做 RL)")
    ap.add_argument("--hard", action="store_true", help="用更难实例(headroom)")
    ap.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="数据集落盘目录（manifest + SMT2）；默认 examples/artifacts/decide_branch_dataset",
    )
    ap.add_argument(
        "--force-regen",
        action="store_true",
        help="忽略已有数据集，按参数重新生成并覆盖落盘",
    )
    ap.add_argument(
        "--no-save-dataset",
        action="store_true",
        help="新生成时不落盘（仅当无已有数据集时有效）",
    )
    ap.add_argument("--z3-path", default=None, help="z3 可执行文件路径（默认同 PATH）")
    ap.add_argument(
        "--binary-timeout",
        type=int,
        default=1200,
        help="（仅提示用）binary 求解请走 solve_dataset_binary --timeout",
    )
    ap.add_argument(
        "--test-workers",
        type=int,
        default=DEFAULT_TEST_WORKERS,
        help=f"测试与 look-ahead 标签构建并发数（默认 {DEFAULT_TEST_WORKERS}）",
    )
    ap.add_argument(
        "--rl-workers",
        type=int,
        default=DEFAULT_RL_COLLECT_WORKERS,
        help="RL collect 进程数（默认 4；实例数<8 时自动串行；与 --test-workers 独立）",
    )
    ap.add_argument(
        "--ckpt-dir",
        default=DEFAULT_CKPT_DIR,
        help="RL 中间 checkpoint 目录",
    )
    ap.add_argument(
        "--ckpt-every",
        type=int,
        default=1,
        help="每隔多少 RL 轮保存一次中间权重（默认每轮）",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="GNN 设备（默认 cuda 可用则 cuda，否则 cpu）",
    )
    args = ap.parse_args()

    device = args.device or gnn_device()
    print(f"GNN device: {device}")

    z3_path = args.z3_path or shutil.which("z3")
    if not z3_path:
        raise SystemExit("未找到 z3 二进制，请用 --z3-path 指定")

    from omt_branching.solver import generate_hard_bool_lia_dataset

    gen = generate_hard_bool_lia_dataset if args.hard else generate_bool_lia_dataset

    torch.manual_seed(0)
    use_existing = dataset_exists(args.dataset_dir) and not args.force_regen
    manifest: dict | None = None
    test_entries: list[dict] = []
    train_entries: list[dict] = []
    train_insts: list[OMTInstance] = []

    if use_existing:
        print(f"使用已有数据集（smt2_to_instance）: {args.dataset_dir}")
        _sync_manifest_if_needed(args.dataset_dir)
        with open(Path(args.dataset_dir) / "manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        test_entries = list_split_entries(args.dataset_dir, "test")
        if not test_entries:
            raise SystemExit(f"数据集无 test 划分: {args.dataset_dir}")
        insts = load_dataset(args.dataset_dir, split="test")
        print(f"测试集 {len(insts)} 个实例已从磁盘加载")
        if args.train > 0 or args.rl_iters > 0:
            train_entries = list_split_entries(args.dataset_dir, "train")
            if not train_entries:
                raise SystemExit(
                    "需要 train 划分但数据集中没有；请用 --force-regen 重新生成，"
                    "或补全 dataset 的 train/"
                )
            train_insts = load_dataset(args.dataset_dir, split="train")
            print(f"训练集 {len(train_insts)} 个实例已从磁盘加载")
    else:
        insts = gen(args.test, seed=99, min_vars=args.min_vars, max_vars=args.max_vars)
        if args.no_save_dataset:
            raise SystemExit(
                "评测/训练 worker 需要落盘 .smt2；请去掉 --no-save-dataset，"
                "或先准备好 --dataset-dir 中的已有数据集。"
            )
        os.makedirs(args.dataset_dir, exist_ok=True)
        gen_name = "hard_bool_lia" if args.hard else "bool_lia"
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generator": gen_name,
            "params": {
                "test": args.test,
                "train": args.train,
                "min_vars": args.min_vars,
                "max_vars": args.max_vars,
                "hard": args.hard,
                "refocus": args.refocus,
                "epochs": args.epochs,
                "rl_iters": args.rl_iters,
            },
            "seeds": {"test": 99, "train": 1},
            "splits": {},
        }
        test_entries = save_dataset(insts, args.dataset_dir, split="test")
        manifest["splits"]["test"] = test_entries
        print(f"测试集 {len(insts)} 个实例已保存 -> {args.dataset_dir}/test/")

        if args.train > 0 or args.rl_iters > 0:
            n_train = args.train if args.train > 0 else 40
            train_insts = gen(
                n_train, seed=1, min_vars=args.min_vars, max_vars=args.max_vars
            )
            train_entries = save_dataset(
                train_insts, args.dataset_dir, split="train"
            )
            manifest["splits"]["train"] = train_entries
            print(
                f"训练集 {len(train_insts)} 个实例已保存 -> {args.dataset_dir}/train/"
            )

        # 直接写入完整 manifest（条目来自刚落盘的实例；save_dataset 已 prune 残留）
        manifest_path = os.path.join(args.dataset_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)
        print(f"数据集清单已保存 -> {manifest_path}")
        print(
            "请先生成 binary 缓存后再评测:\n"
            f"  python -m examples.solve_dataset_binary --dataset-dir {args.dataset_dir}\n"
            "然后再重新运行本脚本（将自动加载已有数据集）。"
        )
        raise SystemExit(0)

    _require_binary_cache(args.dataset_dir, "test", test_entries)

    policy = BranchingPolicy()
    if args.train > 0:
        from omt_branching.model.trainer import ImitationTrainer, TrainConfig
        from omt_branching.solver.training_data import (
            build_lookahead_examples_from_smt2_parallel,
        )

        if not train_insts:
            train_insts = load_dataset(args.dataset_dir, split="train")
        if not train_entries:
            train_entries = list_split_entries(args.dataset_dir, "train")

        lookahead_workers = args.test_workers
        paths = _smt2_abs_paths(args.dataset_dir, train_entries)
        ids = [e["instance_id"] for e in train_entries]
        print(
            f"look-ahead 标签构建: {len(paths)} 实例(from smt2), "
            f"workers={lookahead_workers}"
        )
        exs = [
            e
            for e in build_lookahead_examples_from_smt2_parallel(
                paths,
                instance_ids=ids,
                workers=lookahead_workers,
            )
            if e.bool_target_scores
        ]
        hist = ImitationTrainer(policy, TrainConfig(lr=5e-3, device=device)).fit(
            exs, epochs=args.epochs
        )
        print(
            f"look-ahead imitation: {len(exs)} 样本, branch loss "
            f"{hist[0].get('branch', 0):.3f} -> {hist[-1].get('branch', 0):.3f}"
        )

    if args.rl_iters > 0:
        if not train_insts:
            train_insts = load_dataset(args.dataset_dir, split="train")
        if not train_entries:
            train_entries = list_split_entries(args.dataset_dir, "train")
        rl_train = train_insts
        rl_paths = _smt2_abs_paths(args.dataset_dir, train_entries)
        rl_ids = [e["instance_id"] for e in train_entries]

        rl_workers = effective_rl_workers(len(rl_train), args.rl_workers)
        rlt = DecideRLTrainer(
            policy,
            DecideRLConfig(
                refocus_every=args.refocus,
                device=device,
                workers=rl_workers,
            ),
        )
        mode = f"并行×{rl_workers}" if rl_workers > 1 else "串行(GPU collect)"
        print(
            f"RL collect: {len(rl_train)} 实例 × {args.rl_iters} 轮, {mode} "
            f"(请求 workers={args.rl_workers})；collect 用 CPU，update 用 {device}"
        )
        print(f"RL checkpoints -> {args.ckpt_dir}/ (every {args.ckpt_every})")
        h = rlt.train(
            [i.as_tuple() for i in rl_train],
            iterations=args.rl_iters,
            log=False,
            workers=rl_workers,
            collect_seed=1,
            collect_hard=args.hard,
            collect_min_vars=args.min_vars,
            collect_max_vars=args.max_vars,
            smt2_paths=rl_paths,
            instance_ids=rl_ids,
            checkpoint_dir=args.ckpt_dir,
            checkpoint_every=args.ckpt_every,
        )
        final_path = os.path.join(ARTIFACTS, "rl_decide_policy.pt")
        rlt.save_checkpoint(
            final_path,
            meta={"iter": args.rl_iters, "final": True},
        )
        hist_path = os.path.join(ARTIFACTS, "rl_decide_history.json")
        save_history(h, hist_path)
        print(f"RL 最终权重 -> {final_path}")
        print(f"RL 历史 -> {hist_path}")
        if h:
            print(
                f"RL 微调: {len(h)} 步, 末条 reward={h[-1]['reward']:.3f} "
                f"conflicts={h[-1]['conflicts']}, defer_logit={float(rlt.defer_logit):.3f}"
            )

    agg = {
        "binary": {"rlimit": 0.0, "time_ms": 0.0},
        "vsids": {
            "rlimit": 0.0,
            "decider factory rlimit": 0.0,
            "model base rlimit": 0.0,
            "model cut rlimit": 0.0,
            "check rlimit": 0.0,
            "eval rlimit": 0.0,
            "weighted rlimit": 0.0,
            "conflicts": 0.0,
            "match": 0.0,
        },
        "learned": {
            "rlimit": 0.0,
            "decider factory rlimit": 0.0,
            "model base rlimit": 0.0,
            "model cut rlimit": 0.0,
            "check rlimit": 0.0,
            "eval rlimit": 0.0,
            "weighted rlimit": 0.0,
            "conflicts": 0.0,
            "decisions": 0.0,
            "match": 0.0,
        },
    }
    per_instance: list[dict] = []
    rows = _run_test_parallel(
        test_entries,
        args.dataset_dir,
        policy,
        device,
        args.refocus,
        args.test_workers,
    )
    for row in rows:
        ref_val = row["ref_val"]
        ref = row["binary"]
        v = row["vsids"]
        ln = row["learned"]
        agg["binary"]["rlimit"] += ref.get("rlimit") or 0
        agg["binary"]["time_ms"] += ref.get("time_ms") or 0.0
        for key in v.keys():
            if key not in agg["vsids"]:
                continue
            agg["vsids"][key] += v[key]
        agg["vsids"]["match"] += 1.0 if v["value"] == ref_val else 0.0
        for key in ln.keys():
            if key not in agg["learned"]:
                continue
            agg["learned"][key] += ln[key]
        agg["learned"]["match"] += 1.0 if ln["value"] == ref_val else 0.0
        per_instance.append({
            "instance_id": row["instance_id"],
            "binary": _stats_for_json(ref),
            "vsids": _stats_for_json(v),
            "learned": _stats_for_json(ln),
        })

    n = max(1, len(insts))
    print(
        f"=== 三臂对比（{len(insts)} 实例；binary 为缓存参考；"
        f"match=1 为与 binary 最优值一致）==="
    )
    for key in agg["binary"].keys():
        agg["binary"][key] /= n
    for key in agg["vsids"].keys():
        agg["vsids"][key] /= n
    for key in agg["learned"].keys():
        agg["learned"][key] /= n

    os.makedirs(ARTIFACTS, exist_ok=True)
    results = {
        "reference": "binary_cache",
        "dataset_dir": args.dataset_dir,
        "summary": agg,
        "n_instances": len(insts),
        "z3_path": z3_path,
        "device": device,
        "test_workers": args.test_workers,
        "per_instance": per_instance,
    }
    results_path = os.path.join(ARTIFACTS, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"实验汇总已保存 -> {results_path}")


if __name__ == "__main__":
    main()
