"""面向目标值的 look-ahead 教师（imitation）与缓存构建。

算法（根状态）：
1. 与 ``decide_omt.solve_omt_with_decider`` 相同，经 ``prepare_propagator_formula``
   得到会注册到 UserPropagator 的析取子句原子；
2. 用 z3 二进制求原问题（预处理后硬约束）的目标最优值；
3. 对每个注册原子分别加真/假硬约束，再用 z3 二进制求局部最优；
4. 打分：
   - 真/假均不改变全局最优 → 得分 ``-1``（相位任意）；
   - 一侧全局最优、另一侧 unsat → 得分 ``0``（相位取全局最优侧）；
   - 其余：一侧全局最优、另一侧为更差的局部最优，按
     ``|局部最优 - 全局最优|`` 从大到小排序，得分从 ``1`` 起按名次递增
     （最差 impactful 为 1，最优为 ``m``，步长 1；相位取全局最优侧）；
5. **defer 分**：``max(0, max_score - 2)``，与原子分一并写入样本（供 ListNet 含 defer）。

非根采样：与 defer 同一门槛——固定根上**所有**得分 ``>= defer_score`` 的原子；
``check`` 取 model → Better-cut → 再对剩余原子**剪枝重评**（图断言含 cut）：

- 正分原子：根上最优侧在 cut 后不变，只重解非最优侧；
- 得分 0：两侧不变，保持 0；
- 得分 ``-1``：可能一侧变为局部最优/unsat；若已发现一侧非（局部）全局最优，
  则另一侧必为最优，无需再搜。

缓存布局（与 split look-ahead 分离）::

    lookahead_objective/<split>/<instance_id>.json

运行::

    python -m examples.build_objective_lookahead
    python -m examples.build_objective_lookahead --split train --workers 8 --force
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from omt_branching.input.graph_builder import DEFAULT_FEATURE_SPEC, GraphBuilder
from omt_branching.interfaces import NodeType
from omt_branching.model.trainer import RankingExample
from omt_branching.solver.decide_omt import (
    list_split_entries,
    smt2_to_instance,
    solve_binary,
)
from omt_branching.solver.interfaces import Sense
from omt_branching.solver.instance_gen import OMTInstance
from omt_branching.solver.propagator_snapshot import (
    atom_key,
    build_bool_snapshot,
    prepare_propagator_formula,
)

import z3

ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")
DEFAULT_DATASET_DIR = os.path.join(ARTIFACTS, "dataset")
DEFAULT_WORKERS = max(1, min(12, (os.cpu_count() or 4)))
LOOKAHEAD_OBJECTIVE_SUBDIR = "lookahead_objective"


def objective_lookahead_path(
    dataset_dir,
    instance_id: str,
    *,
    split: str,
) -> Path:
    return Path(dataset_dir) / LOOKAHEAD_OBJECTIVE_SUBDIR / split / f"{instance_id}.json"


def has_objective_lookahead_result(dataset_dir, instance_id: str, *, split: str) -> bool:
    return objective_lookahead_path(dataset_dir, instance_id, split=split).is_file()


def save_objective_lookahead_result(
    dataset_dir,
    instance_id: str,
    *,
    split: str,
    scores: dict[str, float],
    phases: dict[str, bool],
    opt_value,
    n_atoms: int,
    nonroot: dict | None = None,
    defer_score: float | None = None,
) -> Path:
    path = objective_lookahead_path(dataset_dir, instance_id, split=split)
    path.parent.mkdir(parents=True, exist_ok=True)
    dscore = (
        float(defer_score)
        if defer_score is not None
        else _defer_score_from_scores(scores)
    )
    payload = {
        "instance_id": instance_id,
        "split": split,
        "kind": "objective",
        "version": 6,
        "n_atoms": int(n_atoms),
        "opt_value": str(opt_value) if opt_value is not None else None,
        "scores": {str(k): float(v) for k, v in scores.items()},
        "phases": {str(k): bool(v) for k, v in phases.items()},
        "defer_score": dscore,
        "nonroot": None,
    }
    if nonroot is not None:
        nr_scores = nonroot.get("scores") or {}
        payload["nonroot"] = {
            "assignment": {
                str(k): bool(v) for k, v in (nonroot.get("assignment") or {}).items()
            },
            "scores": {str(k): float(v) for k, v in nr_scores.items()},
            "phases": {
                str(k): bool(v) for k, v in (nonroot.get("phases") or {}).items()
            },
            "n_atoms": int(nonroot.get("n_atoms") or 0),
            "n_fixed": int(
                nonroot.get("n_fixed")
                if nonroot.get("n_fixed") is not None
                else len(nonroot.get("assignment") or {})
            ),
            "opt_value": (
                str(nonroot["opt_value"])
                if nonroot.get("opt_value") is not None
                else None
            ),
            "defer_score": float(
                nonroot["defer_score"]
                if nonroot.get("defer_score") is not None
                else _defer_score_from_scores(nr_scores)
            ),
            "model_value": (
                str(nonroot["model_value"])
                if nonroot.get("model_value") is not None
                else None
            ),
        }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def load_objective_lookahead_result(
    dataset_dir,
    instance_id: str,
    *,
    split: str,
) -> Optional[dict]:
    path = objective_lookahead_path(dataset_dir, instance_id, split=split)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("kind") not in (None, "objective"):
        return None
    scores = {str(k): float(v) for k, v in (payload.get("scores") or {}).items()}
    defer = payload.get("defer_score")
    if defer is None:
        defer = _defer_score_from_scores(scores)
    nonroot_raw = payload.get("nonroot")
    nonroot = None
    if isinstance(nonroot_raw, dict) and nonroot_raw.get("scores"):
        nr_scores = {
            str(k): float(v) for k, v in (nonroot_raw.get("scores") or {}).items()
        }
        nr_defer = nonroot_raw.get("defer_score")
        if nr_defer is None:
            nr_defer = _defer_score_from_scores(nr_scores)
        nonroot = {
            "assignment": {
                str(k): bool(v)
                for k, v in (nonroot_raw.get("assignment") or {}).items()
            },
            "scores": nr_scores,
            "phases": {
                str(k): bool(v) for k, v in (nonroot_raw.get("phases") or {}).items()
            },
            "n_atoms": int(nonroot_raw.get("n_atoms") or 0),
            "n_fixed": int(
                nonroot_raw.get("n_fixed")
                if nonroot_raw.get("n_fixed") is not None
                else len(nonroot_raw.get("assignment") or {})
            ),
            "opt_value": nonroot_raw.get("opt_value"),
            "defer_score": float(nr_defer),
            "model_value": nonroot_raw.get("model_value"),
        }
    return {
        "scores": scores,
        "phases": {str(k): bool(v) for k, v in (payload.get("phases") or {}).items()},
        "opt_value": payload.get("opt_value"),
        "n_atoms": payload.get("n_atoms"),
        "defer_score": float(defer),
        "nonroot": nonroot,
        "version": payload.get("version"),
    }


def _defer_score_from_scores(scores: dict[str, float]) -> float:
    """defer 教师分：``max(0, max_score - 2)``；无原子分时为 0。"""
    if not scores:
        return 0.0
    return float(max(0.0, max(scores.values()) - 2.0))


def _z3_num(ref) -> Fraction:
    if z3.is_int_value(ref):
        return Fraction(ref.as_long())
    if z3.is_rational_value(ref):
        return Fraction(ref.numerator_as_long(), ref.denominator_as_long())
    return Fraction(str(ref))


def _check_model_better_cut(
    hard: list,
    objective,
    sense: Sense,
) -> tuple[Optional[Fraction], object | None]:
    """对当前硬约束 ``check`` 取 model，构造 OMT 风格 Better-cut。

    返回 ``(model_value, cut_expr)``；unsat 或无法求值时 ``(None, None)``。
    """
    s = z3.Solver()
    s.add(*hard)
    if s.check() != z3.sat:
        return None, None
    m = s.model()
    val_ref = m.eval(objective, model_completion=True)
    try:
        model_val = _z3_num(val_ref)
    except Exception:
        return None, None
    if sense is Sense.MAX:
        cut = objective > val_ref
    else:
        cut = objective < val_ref
    return model_val, cut


def _as_fraction(v) -> Optional[Fraction]:
    if v is None:
        return None
    if isinstance(v, Fraction):
        return v
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, float):
        return Fraction(v).limit_denominator()
    return Fraction(str(v))


def _forced_instance(inst: OMTInstance, hard_base: list, lit) -> OMTInstance:
    """在硬约束上追加字面量，供 z3 二进制求解。"""
    return replace(
        inst,
        hard=list(hard_base) + [lit],
        instance_id=f"{inst.instance_id}__objla",
    )


def _solve_opt(
    inst: OMTInstance,
    *,
    z3_path: str | None,
    timeout_s: int,
) -> tuple[Optional[Fraction], str]:
    res = solve_binary(inst, z3_path=z3_path, timeout_s=timeout_s)
    status = str(res.get("status") or "error")
    if status != "sat":
        return None, status
    return _as_fraction(res.get("value")), status


def _finalize_atom_scores(
    n: int,
    unaffected: list[str],
    failed_lit: list[tuple[str, bool]],
    impactful: list[tuple[str, Fraction, bool]],
) -> tuple[dict[str, float], dict[str, bool]]:
    """由分类桶生成得分/相位。

    - 无影响：``-1``；
    - 一侧 unsat：``0``；
    - impactful：按 gap 降序，得分 ``m, m-1, ..., 1``（``m=|impactful|``，最优最大）。
    ``n`` 保留以兼容调用方，不再参与赋分。
    """
    del n  # 旧版用 -n / 从 n 递减；现与总原子数解耦
    scores: dict[str, float] = {}
    phases: dict[str, bool] = {}
    for k in unaffected:
        scores[k] = -1.0
        phases[k] = True
    for k, ph in failed_lit:
        scores[k] = 0.0
        phases[k] = ph
    impactful.sort(key=lambda row: (-row[1], row[0]))
    m = len(impactful)
    for i, (k, _gap, ph) in enumerate(impactful):
        scores[k] = float(m - i)  # 最优 → m … 最差 → 1
        phases[k] = ph
    return scores, phases


def _classify_side_pair(
    k: str,
    *,
    opt: Fraction,
    st_t: str,
    val_t: Optional[Fraction],
    st_f: str,
    val_f: Optional[Fraction],
    unaffected: list[str],
    failed_lit: list[tuple[str, bool]],
    impactful: list[tuple[str, Fraction, bool]],
) -> None:
    """把真/假两侧结果写入三个分类桶（跳过 timeout/error）。"""
    if st_t not in ("sat", "unsat") or st_f not in ("sat", "unsat"):
        return
    t_opt = st_t == "sat" and val_t is not None and val_t == opt
    f_opt = st_f == "sat" and val_f is not None and val_f == opt
    t_unsat = st_t == "unsat"
    f_unsat = st_f == "unsat"

    if t_opt and f_opt:
        unaffected.append(k)
        return
    if (t_opt and f_unsat) or (f_opt and t_unsat):
        failed_lit.append((k, True if t_opt else False))
        return
    if t_opt and st_f == "sat" and val_f is not None and val_f != opt:
        impactful.append((k, abs(val_f - opt), True))
        return
    if f_opt and st_t == "sat" and val_t is not None and val_t != opt:
        impactful.append((k, abs(val_t - opt), False))
        return
    if st_t == "sat" and st_f == "sat" and val_t is not None and val_f is not None:
        raise AssertionError("两侧均非全局最优但仍 sat")


def _score_atoms_on_hard(
    inst: OMTInstance,
    hard_base: list,
    atoms: list,
    opt: Fraction,
    *,
    z3_path: str | None = None,
    timeout_s: int = 120,
) -> tuple[dict[str, float], dict[str, bool]]:
    """在给定硬约束上对 ``atoms`` 做目标值 look-ahead 打分（n=len(atoms)）。"""
    n = len(atoms)
    if n == 0:
        return {}, {}

    unaffected: list[str] = []
    failed_lit: list[tuple[str, bool]] = []
    impactful: list[tuple[str, Fraction, bool]] = []

    for a in atoms:
        k = atom_key(a)
        val_t, st_t = _solve_opt(
            _forced_instance(inst, hard_base, a),
            z3_path=z3_path,
            timeout_s=timeout_s,
        )
        val_f, st_f = _solve_opt(
            _forced_instance(inst, hard_base, z3.Not(a)),
            z3_path=z3_path,
            timeout_s=timeout_s,
        )
        _classify_side_pair(
            k,
            opt=opt,
            st_t=st_t,
            val_t=val_t,
            st_f=st_f,
            val_f=val_f,
            unaffected=unaffected,
            failed_lit=failed_lit,
            impactful=impactful,
        )

    return _finalize_atom_scores(n, unaffected, failed_lit, impactful)


def _score_rem_atoms_after_cut(
    inst: OMTInstance,
    hard_cut: list,
    rem_keys: list[str],
    key_to_atom: dict,
    root_scores: dict[str, float],
    root_phases: dict[str, bool],
    opt_cut: Fraction,
    *,
    z3_path: str | None = None,
    timeout_s: int = 120,
) -> tuple[dict[str, float], dict[str, bool]]:
    """非根剪枝重评：利用根评分跳过不变的原子侧。

    前提：Better-cut 后仍 sat 且 ``opt_cut`` 可达（调用方已保证）；已固定所有根上
    得分 ``>= defer`` 的原子（取其根最优相位）后，根上「仅一侧最优」的原子其最优侧不变。
    """
    n = len(rem_keys)
    if n == 0:
        return {}, {}

    unaffected: list[str] = []
    failed_lit: list[tuple[str, bool]] = []
    impactful: list[tuple[str, Fraction, bool]] = []

    for k in rem_keys:
        a = key_to_atom[k]
        sc = float(root_scores.get(k, float("nan")))
        if sc != sc:  # NaN：根上无分，跳过
            continue

        # 得分 0：两侧不变
        if sc == 0.0:
            failed_lit.append((k, bool(root_phases.get(k, True))))
            continue

        # 正分：只重解非最优侧；最优侧仍达 opt_cut
        if sc > 0.0:
            opt_ph = bool(root_phases[k])
            non_opt_lit = z3.Not(a) if opt_ph else a
            val_o, st_o = _solve_opt(
                _forced_instance(inst, hard_cut, non_opt_lit),
                z3_path=z3_path,
                timeout_s=timeout_s,
            )
            if st_o not in ("sat", "unsat"):
                continue
            if st_o == "unsat":
                failed_lit.append((k, opt_ph))
            elif val_o is not None and val_o == opt_cut:
                unaffected.append(k)
            elif val_o is not None:
                impactful.append((k, abs(val_o - opt_cut), opt_ph))
            continue

        # 得分 -1：一侧非最优 ⇒ 另一侧必为最优，可剪掉第二次搜索
        val_t, st_t = _solve_opt(
            _forced_instance(inst, hard_cut, a),
            z3_path=z3_path,
            timeout_s=timeout_s,
        )
        if st_t not in ("sat", "unsat"):
            continue
        t_opt = st_t == "sat" and val_t is not None and val_t == opt_cut
        if not t_opt:
            # 假侧必为最优，无需求解
            if st_t == "unsat":
                failed_lit.append((k, False))
            elif val_t is not None:
                impactful.append((k, abs(val_t - opt_cut), False))
            continue

        val_f, st_f = _solve_opt(
            _forced_instance(inst, hard_cut, z3.Not(a)),
            z3_path=z3_path,
            timeout_s=timeout_s,
        )
        _classify_side_pair(
            k,
            opt=opt_cut,
            st_t=st_t,
            val_t=val_t,
            st_f=st_f,
            val_f=val_f,
            unaffected=unaffected,
            failed_lit=failed_lit,
            impactful=impactful,
        )

    return _finalize_atom_scores(n, unaffected, failed_lit, impactful)


def objective_lookahead_scores(
    inst: OMTInstance,
    *,
    z3_path: str | None = None,
    opt_value=None,
    timeout_s: int = 120,
    sample_nonroot: bool = True,
) -> tuple[dict[str, float], dict[str, bool], Fraction | None, int, dict | None]:
    """计算根状态目标值 look-ahead；可选非根重采样。

    返回 ``(scores, phases, global_opt, n_atoms, nonroot_or_none)``。
    ``nonroot`` 形如 ``{assignment, scores, phases, n_atoms, opt_value}``。
    """
    hard_use, _watch, atoms = prepare_propagator_formula(list(inst.hard))
    n = len(atoms)
    if n == 0:
        return {}, {}, None, 0, None

    key_to_atom = {atom_key(a): a for a in atoms}
    base = replace(inst, hard=list(hard_use), instance_id=f"{inst.instance_id}__base")
    opt = _as_fraction(opt_value)
    if opt is None:
        opt, st = _solve_opt(base, z3_path=z3_path, timeout_s=timeout_s)
        if opt is None:
            return {}, {}, None, n, None

    scores, phases = _score_atoms_on_hard(
        inst, hard_use, atoms, opt, z3_path=z3_path, timeout_s=timeout_s
    )
    if not scores:
        return {}, {}, opt, n, None

    nonroot = None
    if sample_nonroot:
        nonroot = _sample_nonroot_scores(
            inst,
            hard_use=hard_use,
            key_to_atom=key_to_atom,
            root_scores=scores,
            root_phases=phases,
            opt=opt,
            z3_path=z3_path,
            timeout_s=timeout_s,
        )
    return scores, phases, opt, n, nonroot


def _keys_above_defer(
    scores: dict[str, float],
    *,
    key_to_atom: dict | None = None,
) -> tuple[list[str], float]:
    """与 defer 统一：返回得分 ``>= defer_score = max(0, max_score - 2)`` 的**全部**原子。

    不截断数量。按分数降序、键名升序稳定排序。
    """
    defer = _defer_score_from_scores(scores)
    keys = [
        k for k, sc in scores.items()
        if sc >= defer and (key_to_atom is None or k in key_to_atom)
    ]
    keys.sort(key=lambda k: (-scores[k], k))
    return keys, defer


def _sample_nonroot_scores(
    inst: OMTInstance,
    *,
    hard_use: list,
    key_to_atom: dict,
    root_scores: dict[str, float],
    root_phases: dict[str, bool],
    opt: Fraction,
    z3_path: str | None,
    timeout_s: int,
) -> dict | None:
    """固定根上所有得分 >= defer 的原子后：check→model→Better-cut，剪枝重评剩余。"""
    fix_keys, root_defer = _keys_above_defer(root_scores, key_to_atom=key_to_atom)
    rem_keys = [k for k in key_to_atom if k not in fix_keys]
    if not fix_keys or not rem_keys:
        return None

    assignment = {k: bool(root_phases[k]) for k in fix_keys if k in root_phases}
    if len(assignment) != len(fix_keys):
        return None
    lits = []
    for k in fix_keys:
        a = key_to_atom[k]
        lits.append(a if assignment[k] else z3.Not(a))
    hard_nr = list(hard_use) + lits

    model_val, cut = _check_model_better_cut(
        hard_nr, inst.objective, inst.sense
    )
    if model_val is None or cut is None:
        return None
    # model 已是全局最优 ⇒ Better-cut 后 unsat，由下方 opt_cut 求解捕获。
    hard_cut = hard_nr + [cut]

    # cut 后仍须可达到（更优）最优；若 model 已是子问题最优则 cut→UNSAT，跳过。
    base_cut = replace(inst, hard=hard_cut, instance_id=f"{inst.instance_id}__nr_cut")
    opt_cut, st = _solve_opt(base_cut, z3_path=z3_path, timeout_s=timeout_s)
    if opt_cut is None or st != "sat":
        return None

    scores_nr, phases_nr = _score_rem_atoms_after_cut(
        inst,
        hard_cut,
        rem_keys,
        key_to_atom,
        root_scores,
        root_phases,
        opt_cut,
        z3_path=z3_path,
        timeout_s=timeout_s,
    )
    if not scores_nr:
        return None
    return {
        "assignment": assignment,
        "scores": scores_nr,
        "phases": phases_nr,
        "n_atoms": len(rem_keys),
        "n_fixed": len(fix_keys),
        "opt_value": opt_cut,
        "defer_score": _defer_score_from_scores(scores_nr),
        "root_defer_score": root_defer,
        "model_value": model_val,
        "hard_for_graph": hard_cut,
    }


def _cut_from_stored_value(objective, sense: Sense, model_val: Fraction):
    """由缓存的 model 目标值重建 Better-cut（与当时 model.eval 结果一致）。"""
    if model_val.denominator == 1:
        ref = z3.IntVal(int(model_val))
    else:
        ref = z3.RealVal(int(model_val.numerator), int(model_val.denominator))
    if sense is Sense.MAX:
        return objective > ref
    return objective < ref


def _nonroot_graph_hard(
    inst: OMTInstance,
    hard_use: list,
    nonroot: dict,
) -> list:
    """从非根缓存的 assignment + model_value 重建含 Better-cut 的图断言。"""
    from omt_branching.solver.propagator_snapshot import collect_clause_atoms

    if nonroot.get("hard_for_graph") is not None:
        return list(nonroot["hard_for_graph"])
    assignment = nonroot.get("assignment") or {}
    key_to_atom = {atom_key(a): a for a in collect_clause_atoms(hard_use)}
    lits = []
    for k, ph in assignment.items():
        a = key_to_atom.get(k)
        if a is None:
            continue
        lits.append(a if ph else z3.Not(a))
    hard_nr = list(hard_use) + lits
    mv = _as_fraction(nonroot.get("model_value"))
    if mv is None:
        return hard_nr
    return hard_nr + [_cut_from_stored_value(inst.objective, inst.sense, mv)]


def _scores_to_example(
    hard_for_graph: list,
    scores: dict[str, float],
    phases: dict[str, bool],
    *,
    assignment: dict[str, bool] | None = None,
    defer_score: float | None = None,
) -> RankingExample | None:
    snap, _amap = build_bool_snapshot(hard_for_graph, assignment=assignment or {})
    graph = GraphBuilder(DEFAULT_FEATURE_SPEC).build(snap)
    bmap = graph.id_maps.get(NodeType.BOOL_VAR, {})
    bts: dict[int, float] = {}
    pts: dict[int, bool] = {}
    for k, sc in scores.items():
        loc = bmap.get(k)
        if loc is not None:
            bts[loc] = sc
    for k, ph in phases.items():
        loc = bmap.get(k)
        if loc is not None:
            pts[loc] = ph
    if not bts:
        return None
    dscore = (
        float(defer_score)
        if defer_score is not None
        else _defer_score_from_scores(scores)
    )
    return RankingExample(
        graph=graph,
        bool_target_scores=bts,
        phase_targets=pts,
        defer_target_score=dscore,
    )


def build_objective_lookahead_examples(
    inst: OMTInstance,
    *,
    z3_path: str | None = None,
    opt_value=None,
    timeout_s: int = 120,
    include_nonroot: bool = True,
    cached: dict | None = None,
) -> list[RankingExample]:
    """单实例 → 根（+可选非根）RankingExample 列表。"""
    hard_use, _watch, _branch = prepare_propagator_formula(list(inst.hard))
    if cached is not None:
        scores = cached["scores"]
        phases = cached["phases"]
        defer_score = cached.get("defer_score")
        if defer_score is None:
            defer_score = _defer_score_from_scores(scores)
        nonroot = cached.get("nonroot") if include_nonroot else None
    else:
        scores, phases, _opt, _n, nonroot = objective_lookahead_scores(
            inst,
            z3_path=z3_path,
            opt_value=opt_value,
            timeout_s=timeout_s,
            sample_nonroot=include_nonroot,
        )
        defer_score = _defer_score_from_scores(scores)
        if not include_nonroot:
            nonroot = None

    out: list[RankingExample] = []
    if scores:
        ex = _scores_to_example(
            hard_use, scores, phases, defer_score=float(defer_score)
        )
        if ex is not None:
            out.append(ex)
    if include_nonroot and nonroot and nonroot.get("scores"):
        hard_nr_graph = _nonroot_graph_hard(inst, hard_use, nonroot)
        nr_defer = nonroot.get("defer_score")
        if nr_defer is None:
            nr_defer = _defer_score_from_scores(nonroot["scores"])
        ex_nr = _scores_to_example(
            hard_nr_graph,
            nonroot["scores"],
            nonroot["phases"],
            assignment=nonroot.get("assignment") or {},
            defer_score=float(nr_defer),
        )
        if ex_nr is not None:
            out.append(ex_nr)
    return out


def _compute_and_maybe_cache(
    inst: OMTInstance,
    *,
    dataset_dir: str | None = None,
    split: str | None = None,
    use_cache: bool = True,
    cache_only: bool = False,
    include_nonroot: bool = True,
    z3_path: str | None = None,
    opt_value=None,
    timeout_s: int = 120,
) -> list[RankingExample]:
    if use_cache and dataset_dir and split and inst.instance_id:
        cached = load_objective_lookahead_result(
            dataset_dir, inst.instance_id, split=split
        )
        if cached is not None and cached["scores"]:
            return build_objective_lookahead_examples(
                inst, include_nonroot=include_nonroot, cached=cached
            )
        if cache_only:
            return []

    if cache_only:
        return []

    scores, phases, opt, n, nonroot = objective_lookahead_scores(
        inst,
        z3_path=z3_path,
        opt_value=opt_value,
        timeout_s=timeout_s,
        sample_nonroot=include_nonroot,
    )
    if use_cache and dataset_dir and split and inst.instance_id and scores:
        save_objective_lookahead_result(
            dataset_dir,
            inst.instance_id,
            split=split,
            scores=scores,
            phases=phases,
            opt_value=opt,
            n_atoms=n,
            nonroot=nonroot,
            defer_score=_defer_score_from_scores(scores),
        )
    if not scores:
        return []
    return build_objective_lookahead_examples(
        inst,
        include_nonroot=include_nonroot,
        cached={
            "scores": scores,
            "phases": phases,
            "defer_score": _defer_score_from_scores(scores),
            "nonroot": nonroot,
        },
    )


def _from_smt2_worker(task: tuple) -> tuple[int, list[RankingExample]]:
    (
        index,
        smt2_path,
        instance_id,
        dataset_dir,
        split,
        use_cache,
        cache_only,
        include_nonroot,
        z3_path,
        timeout_s,
        opt_value_str,
    ) = task
    inst = smt2_to_instance(smt2_path, instance_id=instance_id)
    opt = _as_fraction(opt_value_str) if opt_value_str is not None else None
    return index, _compute_and_maybe_cache(
        inst,
        dataset_dir=dataset_dir,
        split=split,
        use_cache=use_cache,
        cache_only=cache_only,
        include_nonroot=include_nonroot,
        z3_path=z3_path,
        opt_value=opt,
        timeout_s=timeout_s,
    )


def build_objective_lookahead_examples_from_smt2_parallel(
    smt2_paths: list[str],
    *,
    instance_ids: list[str] | None = None,
    workers: int = DEFAULT_WORKERS,
    dataset_dir: str | None = None,
    split: str | None = None,
    use_cache: bool = True,
    cache_only: bool = False,
    include_nonroot: bool = True,
    z3_path: str | None = None,
    opt_values: list | None = None,
    timeout_s: int = 120,
) -> list[RankingExample]:
    """从已落盘 ``.smt2`` 并行构造目标值 look-ahead imitation 样本。

    ``cache_only=True`` 时只读缓存，缺失则跳过该实例（不现算）。
    ``include_nonroot=True`` 时额外纳入非根样本（若缓存/计算中有）。
    """
    if not smt2_paths:
        return []
    ids = instance_ids or [None] * len(smt2_paths)
    if len(ids) != len(smt2_paths):
        raise ValueError("instance_ids 长度必须与 smt2_paths 一致")
    opts = opt_values if opt_values is not None else [None] * len(smt2_paths)
    if len(opts) != len(smt2_paths):
        raise ValueError("opt_values 长度必须与 smt2_paths 一致")

    if workers <= 1:
        out: list[RankingExample] = []
        for path, iid, ov in zip(smt2_paths, ids, opts):
            inst = smt2_to_instance(path, instance_id=iid)
            out.extend(
                _compute_and_maybe_cache(
                    inst,
                    dataset_dir=dataset_dir,
                    split=split,
                    use_cache=use_cache,
                    cache_only=cache_only,
                    include_nonroot=include_nonroot,
                    z3_path=z3_path,
                    opt_value=ov,
                    timeout_s=timeout_s,
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
            dataset_dir,
            split,
            use_cache,
            cache_only,
            include_nonroot,
            z3_path,
            timeout_s,
            None if opts[i] is None else str(opts[i]),
        )
        for i in range(n)
    ]
    slots: list[list[RankingExample]] = [[] for _ in range(n)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_from_smt2_worker, t) for t in tasks]
        with tqdm(total=len(tasks), desc="obj-lookahead") as pbar:
            for fut in as_completed(futures):
                index, exs = fut.result()
                slots[index] = exs
                pbar.update(1)
    out: list[RankingExample] = []
    for exs in slots:
        out.extend(exs)
    return out


def _cache_worker(task: tuple) -> dict:
    (
        dataset_dir,
        split,
        instance_id,
        smt2_relpath,
        force,
        z3_path,
        timeout_s,
        opt_value_str,
        sample_nonroot,
    ) = task
    if not force and has_objective_lookahead_result(
        dataset_dir, instance_id, split=split
    ):
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
    opt = _as_fraction(opt_value_str) if opt_value_str is not None else None
    scores, phases, opt_v, n, nonroot = objective_lookahead_scores(
        inst,
        z3_path=z3_path,
        opt_value=opt,
        timeout_s=timeout_s,
        sample_nonroot=sample_nonroot,
    )
    if not scores:
        return {
            "instance_id": instance_id,
            "split": split,
            "skipped": False,
            "status": "empty",
            "n_scores": 0,
            "n_atoms": n,
            "has_nonroot": False,
        }
    save_objective_lookahead_result(
        dataset_dir,
        instance_id,
        split=split,
        scores=scores,
        phases=phases,
        opt_value=opt_v,
        n_atoms=n,
        nonroot=nonroot,
    )
    return {
        "instance_id": instance_id,
        "split": split,
        "skipped": False,
        "status": "ok",
        "n_scores": len(scores),
        "n_atoms": n,
        "has_nonroot": nonroot is not None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="并行构建目标值 look-ahead 标签缓存")
    ap.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    ap.add_argument("--split", default=None, help="只处理某一划分（默认全部）")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--z3-path", default=None, help="z3 可执行文件路径")
    ap.add_argument("--timeout", type=int, default=120, help="单次 z3binary 超时（秒）")
    ap.add_argument("--force", action="store_true", help="覆盖已有缓存")
    ap.add_argument(
        "--use-ref-value",
        action="store_true",
        help="若存在 ref 缓存则复用其 binary 最优值作为全局最优（仍对真/假侧跑 z3binary）",
    )
    ap.add_argument(
        "--no-nonroot",
        action="store_true",
        help="不采样非根状态（默认会采样并写入 nonroot 字段）",
    )
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    manifest_path = root / "manifest.json"
    splits: dict[str, list] = {}
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            splits = json.load(f).get("splits", {})
    keys = [args.split] if args.split else (list(splits.keys()) or ["test", "train"])

    ref_values: dict[tuple[str, str], str] = {}
    if args.use_ref_value:
        from omt_branching.solver.binary_results import binary_value, load_binary_result

        for sp in keys:
            entries = splits.get(sp) or list_split_entries(args.dataset_dir, sp)
            for e in entries:
                iid = e["instance_id"]
                if load_binary_result(args.dataset_dir, iid, split=sp) is None:
                    continue
                try:
                    v = binary_value(args.dataset_dir, iid, split=sp)
                except Exception:
                    continue
                if v is not None:
                    ref_values[(sp, iid)] = str(v)

    sample_nonroot = not args.no_nonroot
    tasks: list[tuple] = []
    for sp in keys:
        entries = splits.get(sp) or list_split_entries(args.dataset_dir, sp)
        for e in entries:
            iid = e["instance_id"]
            tasks.append(
                (
                    args.dataset_dir,
                    sp,
                    iid,
                    e["smt2"],
                    args.force,
                    args.z3_path,
                    args.timeout,
                    ref_values.get((sp, iid)),
                    sample_nonroot,
                )
            )

    if not tasks:
        print("无可处理实例")
        return

    n_workers = max(1, min(args.workers, len(tasks)))
    print(
        f"objective lookahead 缓存: {len(tasks)} 实例, workers={n_workers}, "
        f"force={args.force}, use_ref_value={args.use_ref_value}, "
        f"nonroot={sample_nonroot}"
    )
    print(f"结果目录: {root / LOOKAHEAD_OBJECTIVE_SUBDIR}/")

    stats = {"cached": 0, "ok": 0, "empty": 0, "fail": 0, "nonroot": 0}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_cache_worker, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="obj-lookahead") as pbar:
            for fut in as_completed(futures):
                try:
                    row = fut.result()
                except Exception:
                    stats["fail"] += 1
                    pbar.set_postfix(**stats)
                    pbar.update(1)
                    continue
                st = row.get("status")
                if row.get("skipped"):
                    stats["cached"] += 1
                elif st == "ok":
                    stats["ok"] += 1
                    if row.get("has_nonroot"):
                        stats["nonroot"] += 1
                elif st == "empty":
                    stats["empty"] += 1
                else:
                    stats["fail"] += 1
                pbar.set_postfix(**stats)
                pbar.update(1)

    print(
        f"完成: cached={stats['cached']} ok={stats['ok']} "
        f"nonroot={stats['nonroot']} empty={stats['empty']} fail={stats['fail']}"
    )


if __name__ == "__main__":
    main()
