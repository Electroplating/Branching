"""SMT2 ↔ OMTInstance 往返测试。

验证新增的**读**接口（``smt2_to_instance`` / ``load_dataset``）与既有**写**接口
（``instance_to_smt2`` / 数据集落盘）互逆：往返后同一 ``solve_native`` 最优值不变，
方向/变量/目标系数一致。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

z3 = pytest.importorskip("z3")

from omt_branching.solver import (
    generate_instance,
    generate_lra_instance,
    instance_to_smt2,
    load_dataset,
    smt2_to_instance,
    solve_native,
)
from omt_branching.solver.interfaces import Sense


def _assert_same_instance(inst, inst2):
    """往返后：最优值 / 方向 / 变量名 / 目标系数 / 理论一致。"""
    v1 = solve_native(*inst.as_tuple())["value"]
    v2 = solve_native(*inst2.as_tuple())["value"]
    assert v1 is not None and v1 == v2
    assert inst2.sense is inst.sense
    assert [v.decl().name() for v in inst2.variables] == [
        v.decl().name() for v in inst.variables
    ]
    assert inst2.obj_coeffs == inst.obj_coeffs
    assert inst2.theory == inst.theory


def test_smt2_to_instance_roundtrips_lia_box():
    inst = generate_instance(
        "t0", random.Random(0), n_vars=4, n_constraints=6, ub=8,
        max_coeff=5, sense=Sense.MAX,
    )
    inst2 = smt2_to_instance(instance_to_smt2(inst))
    _assert_same_instance(inst, inst2)


def test_smt2_to_instance_roundtrips_lra_negative_coeffs():
    # LRA 目标系数可为负/零：往返必须精确还原。
    inst = generate_lra_instance("lra0", random.Random(3), n_vars=4, family="mixed")
    inst2 = smt2_to_instance(instance_to_smt2(inst))
    _assert_same_instance(inst, inst2)


def test_smt2_to_instance_reads_from_file(tmp_path):
    inst = generate_instance("f0", random.Random(1), n_vars=3, sense=Sense.MIN)
    p = tmp_path / "f0.smt2"
    p.write_text(instance_to_smt2(inst), encoding="utf-8")
    inst2 = smt2_to_instance(p)
    assert inst2.instance_id == "f0"  # 无显式 id 时取文件名
    _assert_same_instance(inst, inst2)


def test_load_dataset_roundtrips(tmp_path):
    insts = [
        generate_instance(f"d{i}", random.Random(i), n_vars=3, sense=Sense.MAX)
        for i in range(3)
    ]
    split = "train"
    (tmp_path / split).mkdir(parents=True, exist_ok=True)
    entries = []
    for inst in insts:
        rel = f"{split}/{inst.instance_id}.smt2"
        (tmp_path / rel).write_text(instance_to_smt2(inst), encoding="utf-8")
        entries.append({
            "instance_id": inst.instance_id, "theory": inst.theory,
            "family": inst.family, "description": inst.description,
            "sense": inst.sense.value, "obj_coeffs": inst.obj_coeffs, "smt2": rel,
        })
    (tmp_path / "manifest.json").write_text(
        json.dumps({"splits": {split: entries}}), encoding="utf-8"
    )

    loaded = load_dataset(tmp_path)
    assert len(loaded) == len(insts)
    by_id = {x.instance_id: x for x in loaded}
    for inst in insts:
        got = by_id[inst.instance_id]
        assert got.family == inst.family          # 元数据经 manifest 还原
        _assert_same_instance(inst, got)


def test_missing_objective_raises():
    smt2 = "(set-logic QF_LIA)\n(declare-fun x () Int)\n(assert (>= x 0))\n(check-sat)\n"
    with pytest.raises(ValueError):
        smt2_to_instance(smt2)


def test_multiple_objectives_raises():
    smt2 = (
        "(set-logic QF_LIA)\n(declare-fun x () Int)\n(declare-fun y () Int)\n"
        "(assert (>= x 0))\n(maximize x)\n(minimize y)\n(check-sat)\n"
    )
    with pytest.raises(NotImplementedError):
        smt2_to_instance(smt2)
