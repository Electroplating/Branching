"""加载 checkpoint + 单个 SMT2，按 ``solve_omt_with_decider`` 全量求解（无 ref_rlimit 剪枝），
并在 propagator ``_on_decide`` 处实时把决策原子写入日志。

运行::

    python -m examples.checkpoint_trace \\
        --ckpt examples/artifacts/rl_decide_policy.pt \\
        --smt2 path/to/inst.smt2 \\
        --log examples/artifacts/checkpoint_trace.log

日志每行一条 decide 回调（立即 flush）：
``seq=<n> source=gnn|vsids atom=<key> phase=True|False n_undecided=<m>``
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import z3

from omt_branching.model.device import gnn_device
from omt_branching.model.persistence import load_policy
from omt_branching.solver.decide_omt import smt2_to_instance, solve_omt_with_decider
from omt_branching.solver.propagator import LearnedDecidePropagator
from omt_branching.solver.rl_decide import SamplingPolicyDecider

ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")
DEFAULT_CKPT = os.path.join(ARTIFACTS, "rl_decide_policy.pt")
DEFAULT_LOG = os.path.join(ARTIFACTS, "checkpoint_trace.log")


class TracingDecidePropagator(LearnedDecidePropagator):
    """在 ``_on_decide`` 中记录并实时写出决策原子；计数走父类共享 ``_counters``。"""

    def __init__(self, s, atoms, decider, *, log_fp, seq_holder: list, _counters=None):
        super().__init__(s, atoms, decider, _counters=_counters)
        self._log_fp = log_fp
        # 与 fresh() 副本共享同一计数器（单元素 list）
        self._seq_holder = seq_holder

    def fresh(self, new_ctx):
        return TracingDecidePropagator(
            new_ctx,
            self.atoms,
            self.decider,
            log_fp=self._log_fp,
            seq_holder=self._seq_holder,
            _counters=self._counters,
        )

    def _emit(self, *, source: str, atom: str, phase: bool, n_undecided: int) -> None:
        self._seq_holder[0] += 1
        line = (
            f"seq={self._seq_holder[0]} source={source} atom={atom} "
            f"phase={phase} n_undecided={n_undecided}\n"
        )
        self._log_fp.write(line)
        self._log_fp.flush()

    def _on_decide(self, t, idx, phase):
        self._counters.on_decide += 1
        undecided = [k for k in self.key2atom if k not in self._val]
        if not undecided:
            self._counters.empty += 1
            return
        choice = self.decider(undecided, self._val)
        if choice is None:
            self._counters.defer += 1
            # 退回 VSIDS：记录 z3 拟分裂的文字（若落在已注册原子上）
            vsids_key = self._id2key.get(t.get_id())
            atom_s = vsids_key if vsids_key is not None else t.sexpr()
            ph = int(phase) == int(z3.Z3_L_TRUE)
            self._emit(
                source="vsids",
                atom=atom_s,
                phase=ph,
                n_undecided=len(undecided),
            )
            return
        key, ph = choice
        atom = self.key2atom.get(key)
        if atom is None:
            self._counters.bad_key += 1
            return
        self._emit(
            source="gnn",
            atom=str(key),
            phase=bool(ph),
            n_undecided=len(undecided),
        )
        self._counters.next_split += 1
        z3_phase = z3.Z3_L_TRUE if ph else z3.Z3_L_FALSE
        self.next_split(atom, 0, z3_phase)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="checkpoint + SMT2：无剪枝 OMT 求解，并实时记录 decide 原子"
    )
    ap.add_argument(
        "--ckpt",
        "--checkpoint",
        dest="checkpoint",
        default=DEFAULT_CKPT,
        help=f"策略权重 .pt（默认 {DEFAULT_CKPT}）",
    )
    ap.add_argument(
        "--smt2",
        required=True,
        help="待测单目标 OMT 的 .smt2 路径",
    )
    ap.add_argument(
        "--log",
        default=DEFAULT_LOG,
        help=f"decide 轨迹日志路径（默认 {DEFAULT_LOG}）",
    )
    ap.add_argument("--refocus", type=int, default=50)
    ap.add_argument(
        "--sticky-window",
        action="store_true",
        help="启用窗口粘性（默认关闭）",
    )
    ap.add_argument(
        "--defer-logit",
        type=float,
        default=None,
        help="覆盖 checkpoint meta 中的 defer_logit（默认读 meta，否则 0）",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="GNN 设备（默认 cuda 可用则 cuda，否则 cpu）",
    )
    ap.add_argument(
        "--max-iters",
        type=int,
        default=100000,
        help="OMT better-cut 最大轮数（与 solve_omt_with_decider 默认一致）",
    )
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    smt2_path = Path(args.smt2)
    log_path = Path(args.log)
    if not ckpt_path.is_file():
        raise SystemExit(f"权重不存在: {ckpt_path}")
    if not smt2_path.is_file():
        raise SystemExit(f"SMT2 不存在: {smt2_path}")

    device = args.device or gnn_device()
    policy, meta = load_policy(ckpt_path, map_location="cpu")
    policy.to(device)
    policy.eval()
    if args.defer_logit is not None:
        defer_logit = float(args.defer_logit)
    else:
        defer_logit = float(meta.get("defer_logit", 0.0)) if meta else 0.0
    sticky_window = bool(args.sticky_window)

    inst = smt2_to_instance(smt2_path)
    hard, obj, sense = inst.as_tuple()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    seq_holder = [0]
    print(f"GNN device: {device}")
    print(f"checkpoint: {ckpt_path.resolve()}")
    print(f"smt2: {smt2_path.resolve()} (id={inst.instance_id})")
    print(f"defer_logit={defer_logit}, sticky_window={sticky_window}, refocus={args.refocus}")
    print(f"decide 日志: {log_path.resolve()}（无 ref_rlimit 剪枝）")
    sys.stdout.flush()

    defer = torch.nn.Parameter(
        torch.tensor(float(defer_logit), dtype=torch.float32, device=device)
    )

    def decider_factory(assertions):
        return SamplingPolicyDecider(
            policy,
            defer,
            assertions,
            args.refocus,
            sample=False,
            device=device,
            sticky_window=sticky_window,
        )

    with open(log_path, "w", encoding="utf-8", buffering=1) as log_fp:
        log_fp.write(
            f"# checkpoint_trace instance={inst.instance_id} "
            f"ckpt={ckpt_path} defer_logit={defer_logit}\n"
        )
        log_fp.flush()

        def propagator_factory(s, atoms, decider):
            return TracingDecidePropagator(
                s, atoms, decider, log_fp=log_fp, seq_holder=seq_holder
            )

        stats = solve_omt_with_decider(
            hard,
            obj,
            sense,
            decider_factory=decider_factory,
            max_iters=args.max_iters,
            ref_rlimit=None,
            propagator_factory=propagator_factory,
        )

    print(
        f"完成: value={stats.get('value')} rlimit={stats.get('rlimit')} "
        f"consequence={stats.get('consequence rlimit')} "
        f"rlimit+cons={stats.get('rlimit with consequence')} "
        f"weighted={stats.get('weighted rlimit')} conflicts={stats.get('conflicts')} "
        f"better_cut_iters={stats.get('better_cut_iters')} "
        f"on_decide={stats.get('on_decide')} next_split={stats.get('next_split')} "
        f"defer={stats.get('defer')} "
        f"truncated={stats.get('truncated')} log_lines={seq_holder[0]}"
    )


if __name__ == "__main__":
    main()
