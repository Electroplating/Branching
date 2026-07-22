"""VSIDS 轨迹观察 + 模仿样本：把 z3 原生 VSIDS 的**逐步决策**作 imitation 冷启动标签。

动机：旧 look-ahead 教师(``lookahead.py``)只在**根状态**打一次静态 ``consequences`` 分数,
教出的是准静态全序 —— 在 SMT(LIA) 上输给 VSIDS 的冲突自适应动态。VSIDS 轨迹标签在**真实
搜索中间状态**采集(部分赋值经 ``build_bool_snapshot(assignment=...)`` 编码,与 RL 回路同款),
故能把 VSIDS 的动态行为作为 BC 目标 —— 但只作 **warm-start**:纯模仿至多追平 VSIDS,靠其后
的 REINFORCE(reward=归一化 −conflicts)才可能反超。

机制:z3 的 ``add_decide`` 回调既是 override 钩子也是**观察点** —— 参数 ``t`` 正是 z3 自身
(VSIDS)将要分裂的文字。观察臂在每次 decide 记录 ``(当前赋值, atom_key(t), 相位)`` 后直接
返回(不 ``next_split``),让 VSIDS 照常执行 ``t``,即免费录得 VSIDS 的完整决策轨迹,不改 z3。

已知边界:仅记录 VSIDS 落在**已注册原子**上的决策(未注册的辅助/Tseitin 变量跳过)——
部署 decider 也只在注册原子间选,故这是可获得的最贴近的克隆目标;VSIDS 选辅助变量的状态无
标签(全覆盖 override 与 VSIDS 的固有差,留给 RL 弥合)。
"""
from __future__ import annotations

from dataclasses import dataclass

import z3

from omt_branching.input.graph_builder import DEFAULT_FEATURE_SPEC, GraphBuilder
from omt_branching.interfaces import NodeType
from omt_branching.model.trainer import RankingExample
from omt_branching.solver.propagator_snapshot import atom_key, build_bool_snapshot


@dataclass(frozen=True)
class VSIDSTraceConfig:
    """采集控制。``stride``>1 每 stride 次注册-原子决策留 1 条;``max_examples``>0 为单实例
    上限(0=不限);``weight`` 是 one-hot 目标锐度 —— ListNet 损失会对目标 softmax,故用较大值
    使 softmax 近似 one-hot(见 ``build_vsids_examples_sat``)。
    """

    stride: int = 1
    max_examples: int = 0
    weight: float = 10.0


def _stat(s, key):
    st = s.statistics()
    for k in st.keys():
        if k == key:
            return st.get_key_value(k)
    return 0


class _VSIDSTraceProp(z3.UserPropagateBase):
    """观察-only propagator:跟踪赋值(add_fixed),在每次 decide 记录 VSIDS 的选择后放行。

    与 :class:`LearnedDecidePropagator` 同构：``atoms`` 全量 watch；``branch_atoms`` 才记
    录 VSIDS 决策。``_on_decide`` 不 ``next_split`` —— 即不接管、只观察。
    """

    def __init__(
        self,
        s,
        atoms,
        sink: list,
        config: VSIDSTraceConfig,
        *,
        branch_atoms=None,
    ):
        super().__init__(s)
        self.atoms = list(atoms)
        self.key2atom = {atom_key(a): a for a in self.atoms}
        self._id2key = {a.get_id(): k for k, a in self.key2atom.items()}
        if branch_atoms is None:
            branch_list = list(self.atoms)
        else:
            branch_list = list(branch_atoms)
        self.branch_atoms = branch_list
        self.branch_keys = {atom_key(a) for a in branch_list}
        for a in branch_list:
            k = atom_key(a)
            if k not in self.key2atom:
                self.atoms.append(a)
                self.key2atom[k] = a
                self._id2key[a.get_id()] = k
        self.sink = sink  # (assignment_copy, trail_copy, chosen_key, phase_bool)
        self.config = config
        self._val: dict = {}
        self._trail: list = []
        self._lim: list = []
        self.n_seen = 0  # 落在分支原子上的 VSIDS 决策数
        self.n_records = 0
        self.add_fixed(self._on_fixed)
        self.add_decide(self._on_decide)
        for a in self.atoms:
            self.add(a)

    def push(self):
        self._lim.append(len(self._trail))

    def pop(self, num_scopes):
        for _ in range(num_scopes):
            lim = self._lim.pop()
            while len(self._trail) > lim:
                self._val.pop(self._trail.pop(), None)

    def fresh(self, new_ctx):
        return _VSIDSTraceProp(
            new_ctx,
            self.atoms,
            self.sink,
            self.config,
            branch_atoms=self.branch_atoms,
        )

    def _on_fixed(self, t, v):
        k = self._id2key.get(t.get_id())
        if k is not None and k not in self._val:
            self._val[k] = z3.is_true(v)
            self._trail.append(k)

    def _on_decide(self, t, idx, phase):
        # t = VSIDS 将要分裂的文字。只观察落在分支候选上的决策。
        k = self._id2key.get(t.get_id())
        if k is None or k not in self.branch_keys or k in self._val:
            return  # 非分支 / 未 watch / 已定 -> 跳过
        self.n_seen += 1
        if self.config.stride > 1 and (self.n_seen % self.config.stride) != 0:
            return
        if self.config.max_examples and self.n_records >= self.config.max_examples:
            return
        ph = int(phase) == int(z3.Z3_L_TRUE)
        asg = {kk: self._val[kk] for kk in self._trail}
        self.sink.append((asg, list(self._trail), k, ph))
        self.n_records += 1
        # 不 next_split -> 放行 VSIDS 执行它自己的 t


def collect_vsids_trajectory(
    assertions,
    atoms,
    config: VSIDSTraceConfig = VSIDSTraceConfig(),
    *,
    branch_atoms=None,
):
    """单实例观察 VSIDS 一遍。返回 ``(records, ref_conflicts, info)``:

    - ``records``: ``list[(assignment, trail, chosen_key, phase_bool)]``;
    - ``ref_conflicts``: 本次(纯 VSIDS,观察-only 未覆盖)的 conflicts,作 RL 归一化参考;
    - ``info``: 结果摘要。

    ``atoms`` 为 watch 全量；``branch_atoms`` 默认等于 ``atoms``（纯 SAT）；OMT 应传入
    析取分支集。
    """
    s = z3.Solver()
    sink: list = []
    prop = _VSIDSTraceProp(
        s, list(atoms), sink, config, branch_atoms=branch_atoms
    )
    s.add(*assertions)
    res = s.check()
    ref_conflicts = _stat(s, "conflicts")
    info = {
        "result": "sat" if res == z3.sat else ("unsat" if res == z3.unsat else "unknown"),
        "conflicts": ref_conflicts,
        "decisions_registered": prop.n_seen,
        "records": len(sink),
        "rlimit": _stat(s, "rlimit count"),
    }
    return sink, ref_conflicts, info


def records_to_examples(
    assertions,
    records: list,
    config: VSIDSTraceConfig = VSIDSTraceConfig(),
) -> list[RankingExample]:
    """把轨迹记录转为 RankingExample：每条记录一张图 + 近似 one-hot 标签。"""
    out: list[RankingExample] = []
    hard = list(assertions)
    for rec in records:
        if len(rec) == 4:
            assignment, trail, chosen_key, phase = rec
        else:
            assignment, chosen_key, phase = rec
            trail = list(assignment.keys())
        snap, _ = build_bool_snapshot(hard, assignment=assignment, trail=trail)
        graph = GraphBuilder(DEFAULT_FEATURE_SPEC).build(snap)
        bmap = graph.id_maps.get(NodeType.BOOL_VAR, {})
        if chosen_key not in bmap:
            continue
        undecided = [k for k in bmap if k not in assignment]
        bts = {bmap[k]: (config.weight if k == chosen_key else 0.0) for k in undecided}
        bts[bmap[chosen_key]] = config.weight  # 保证被选原子入表
        pts = {bmap[chosen_key]: phase}
        out.append(RankingExample(graph=graph, bool_target_scores=bts, phase_targets=pts))
    return out


def build_vsids_examples_sat(problems, config: VSIDSTraceConfig = VSIDSTraceConfig()):
    """VSIDS 模仿样本(与 ``build_lookahead_examples_sat`` 同签名,可直接替换教师)。

    ``problems = list[(atoms, clauses)]``。对每个 VSIDS 决策状态建图并打**近似 one-hot**标签:
    ``bool_target_scores`` 里 VSIDS 所选原子记 ``weight``、其余未定原子记 ``0`` —— 经 ListNet
    的 ``softmax(target)`` 后近似 one-hot,损失即"在全部未定原子上预测 VSIDS 之选"的交叉熵。
    """
    out: list[RankingExample] = []
    for atoms, assertions in problems:
        records, _ref, _info = collect_vsids_trajectory(
            list(assertions), list(atoms), config
        )
        out.extend(records_to_examples(assertions, records, config))
    return out


def _omt_assertions_and_atoms(inst):
    """与部署 decider 一致：预处理后断言 + watch 全量 + 析取分支集。"""
    from omt_branching.solver.propagator_snapshot import prepare_propagator_formula

    return prepare_propagator_formula(list(inst.hard))


def _compute_and_maybe_cache_vsids(
    inst,
    config: VSIDSTraceConfig,
    *,
    dataset_dir: str | None = None,
    split: str | None = None,
    use_cache: bool = True,
    cache_only: bool = False,
) -> list[RankingExample]:
    """采集/读缓存 VSIDS 轨迹并建样本。

    ``cache_only=True`` 时只读缓存，缺失则返回空列表（不现算）。
    """
    from omt_branching.solver.vsids_trace_cache import (
        load_vsids_trace_result,
        save_vsids_trace_result,
    )

    assertions, watch, branch = _omt_assertions_and_atoms(inst)
    cached = None
    if use_cache and dataset_dir and split and inst.instance_id:
        cached = load_vsids_trace_result(
            dataset_dir,
            inst.instance_id,
            split=split,
            stride=config.stride,
            max_examples=config.max_examples,
        )
    if cached is not None:
        return records_to_examples(assertions, cached["records"], config)
    if cache_only:
        return []

    records, ref_conflicts, info = collect_vsids_trajectory(
        assertions, watch, config, branch_atoms=branch
    )
    # 空轨迹也落盘：标记「已采集、无分支原子决策」，避免每次被当成缺失重算
    if use_cache and dataset_dir and split and inst.instance_id:
        save_vsids_trace_result(
            dataset_dir,
            inst.instance_id,
            split=split,
            records=records,
            ref_conflicts=ref_conflicts,
            info=info,
            stride=config.stride,
            max_examples=config.max_examples,
        )
    return records_to_examples(assertions, records, config)


def build_vsids_examples(instances, config: VSIDSTraceConfig = VSIDSTraceConfig()):
    """从 OMT 实例构造 VSIDS 轨迹 imitation 样本（与部署注册原子一致）。"""
    out: list[RankingExample] = []
    for inst in instances:
        out.extend(_compute_and_maybe_cache_vsids(inst, config, use_cache=False))
    return out


def _vsids_from_smt2_worker(task: tuple) -> tuple[int, list[RankingExample]]:
    """ProcessPool worker：从已落盘 ``.smt2`` 读实例；优先用 VSIDS 轨迹缓存。"""
    (
        index,
        smt2_path,
        instance_id,
        stride,
        max_examples,
        weight,
        dataset_dir,
        split,
        use_cache,
        cache_only,
    ) = task
    from omt_branching.solver.decide_omt import smt2_to_instance

    inst = smt2_to_instance(smt2_path, instance_id=instance_id)
    cfg = VSIDSTraceConfig(
        stride=stride, max_examples=max_examples, weight=weight
    )
    return index, _compute_and_maybe_cache_vsids(
        inst,
        cfg,
        dataset_dir=dataset_dir,
        split=split,
        use_cache=use_cache,
        cache_only=cache_only,
    )


DEFAULT_VSIDS_WORKERS = 8


def build_vsids_examples_from_smt2_parallel(
    smt2_paths: list[str],
    *,
    instance_ids: list[str] | None = None,
    config: VSIDSTraceConfig | None = None,
    workers: int = DEFAULT_VSIDS_WORKERS,
    dataset_dir: str | None = None,
    split: str | None = None,
    use_cache: bool = True,
    cache_only: bool = False,
) -> list[RankingExample]:
    """从已落盘 ``.smt2`` 并行构造 VSIDS 轨迹样本。

    ``dataset_dir`` + ``split`` 非空且 ``use_cache`` 时：优先读
    ``vsids_trace/<split>/<id>.json``；缺失则采集后即时写入（``cache_only`` 时跳过）。
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from tqdm import tqdm

    if not smt2_paths:
        return []
    cfg = config or VSIDSTraceConfig()
    ids = instance_ids or [None] * len(smt2_paths)
    if len(ids) != len(smt2_paths):
        raise ValueError("instance_ids 长度必须与 smt2_paths 一致")
    if workers <= 1:
        from omt_branching.solver.decide_omt import smt2_to_instance

        out: list[RankingExample] = []
        for path, iid in zip(smt2_paths, ids):
            inst = smt2_to_instance(path, instance_id=iid)
            out.extend(
                _compute_and_maybe_cache_vsids(
                    inst,
                    cfg,
                    dataset_dir=dataset_dir,
                    split=split,
                    use_cache=use_cache,
                    cache_only=cache_only,
                )
            )
        return out

    n = len(smt2_paths)
    workers = min(workers, n)
    tasks = [
        (
            i,
            smt2_paths[i],
            ids[i],
            cfg.stride,
            cfg.max_examples,
            cfg.weight,
            dataset_dir,
            split,
            use_cache,
            cache_only,
        )
        for i in range(n)
    ]
    slots: list[list[RankingExample]] = [[] for _ in range(n)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_vsids_from_smt2_worker, t) for t in tasks]
        with tqdm(total=len(tasks), desc="vsids_trace") as pbar:
            for fut in as_completed(futures):
                index, exs = fut.result()
                slots[index] = exs
                pbar.update(1)
    out: list[RankingExample] = []
    for exs in slots:
        out.extend(exs)
    return out


__all__ = [
    "VSIDSTraceConfig",
    "collect_vsids_trajectory",
    "records_to_examples",
    "build_vsids_examples_sat",
    "build_vsids_examples",
    "build_vsids_examples_from_smt2_parallel",
    "DEFAULT_VSIDS_WORKERS",
]
