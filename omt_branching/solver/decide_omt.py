"""OMT = 单 z3.Solver 线性搜索回路（Solve + Better-cut，直到 UNSAT），可挂
LearnedDecidePropagator 接管内部布尔决策。z3.Optimize 不支持 propagator，故必须走此回路。

对比臂：``solve_native``（Python Optimize API）、``solve_binary``（z3 二进制 ``-st``）、
``solve_omt_with_decider``（VSIDS / learned UserPropagator 回路）。
"""

from __future__ import annotations

from fractions import Fraction
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Optional

import z3

from omt_branching.solver.interfaces import Sense
from omt_branching.solver.propagator import LearnedDecidePropagator
from omt_branching.solver.propagator_snapshot import collect_atoms


def _stat(s, key):
    st = s.statistics()
    for k in st.keys():
        if k == key:
            return st.get_key_value(k)
    return 0


def _num(ref):
    if z3.is_int_value(ref):
        return ref.as_long()
    if z3.is_rational_value(ref):
        return Fraction(ref.numerator_as_long(), ref.denominator_as_long())
    return Fraction(str(ref))


def solve_omt_with_decider(
    hard,
    objective,
    sense: Sense,
    decider_factory=None,
    max_iters: int = 100000,
) -> dict:
    s = z3.Solver()
    solver_rlimit = _stat(s, "rlimit count")
    rlimit = solver_rlimit
    prop = None
    if decider_factory is not None:
        atoms = collect_atoms(list(hard))
        decider = decider_factory(list(hard))
        prop = LearnedDecidePropagator(s, atoms, decider)
    decider_factory_rlimit = _stat(s, "rlimit count") - rlimit
    rlimit += decider_factory_rlimit

    s.add(*hard)
    model_rlimit = [_stat(s, "rlimit count") - rlimit]
    rlimit += model_rlimit[-1]

    if s.check() != z3.sat:
        raise ValueError("硬约束不可满足")
    check_rlimit = [_stat(s, "rlimit count") - rlimit]
    rlimit += check_rlimit[-1]

    m = s.model()
    best_val = m.eval(objective, model_completion=True)
    eval_rlimit = [_stat(s, "rlimit count") - rlimit]
    rlimit += eval_rlimit[-1]

    records = [(_num(best_val), check_rlimit[-1] + eval_rlimit[-1])]

    iters = 0
    for iters in range(1, max_iters + 1):
        cut = objective > best_val if sense is Sense.MAX else objective < best_val
        s.add(cut)
        model_rlimit.append(_stat(s, "rlimit count") - rlimit)
        rlimit += model_rlimit[-1]

        if s.check() != z3.sat:
            break
        check_rlimit.append(_stat(s, "rlimit count") - rlimit)
        rlimit += check_rlimit[-1]

        m = s.model()
        best_val = m.eval(objective, model_completion=True)
        eval_rlimit.append(_stat(s, "rlimit count") - rlimit)
        rlimit += eval_rlimit[-1]

        records.append((_num(best_val), check_rlimit[-1] + eval_rlimit[-1]))

    stats = {
        "value": _num(best_val),
        # "rlimit": _stat(s, "rlimit count"),
        "rlimit": decider_factory_rlimit
        + sum(model_rlimit)
        + sum(check_rlimit)
        + sum(eval_rlimit),
        "conflicts": _stat(s, "conflicts"),
        "decisions": (prop.n_decisions if prop is not None else None),
        "iters": iters,
    }

    local, cost = records[0]
    weighted_rlimit = len(records) * cost
    for i in range(1, len(records)):
        last_local = local
        local, cost = records[i]
        weighted_rlimit += (
            # max(
            #     (stats["value"] - last_local) / (local - last_local),
            #     len(records) - i,
            # )
            (len(records) - i)
            * cost
        )
    stats["weighted rlimit"] = weighted_rlimit

    # stats["solver rlimit"] = solver_rlimit
    stats["decider factory rlimit"] = decider_factory_rlimit
    stats["model base rlimit"] = model_rlimit[0]
    stats["model cut rlimit"] = sum(model_rlimit) - model_rlimit[0]
    stats["check rlimit"] = sum(check_rlimit)
    stats["eval rlimit"] = sum(eval_rlimit)

    return stats


def solve_native(
    hard,
    obj,
    sense: Sense,
    max_rlimit: int = -1,
):
    ctx = z3.Context()
    o = z3.Optimize(ctx=ctx)
    if max_rlimit > 0:
        o.set("rlimit", max_rlimit)
    hard_iso = [h.translate(ctx) for h in hard]
    obj_iso = obj.translate(ctx)
    o.add(*hard_iso)
    if sense is Sense.MIN:
        o.minimize(obj_iso)
    else:
        o.maximize(obj_iso)
    res = o.check()
    rlimit = _stat(o, "rlimit count")
    if res != z3.sat:
        return {
            "value": None,
            "rlimit": rlimit,
        }
    m = o.model()
    return {
        "value": _num(m.eval(obj_iso, model_completion=True)),
        "rlimit": rlimit,
    }


def _collect_vars(hard, obj) -> list:
    """从硬约束与目标中收集算术变量（去重、按名排序）。"""
    seen: dict[str, object] = {}

    def visit(expr) -> None:
        if z3.is_const(expr) and expr.decl().arity() == 0:
            if z3.is_int(expr) or z3.is_real(expr):
                seen[str(expr)] = expr
        for child in expr.children():
            visit(child)

    for h in hard:
        visit(h)
    visit(obj)
    return [seen[k] for k in sorted(seen)]


def _has_or(expr) -> bool:
    if z3.is_or(expr):
        return True
    return any(_has_or(c) for c in expr.children())


def _smt_logic_and_sort(hard, variables: list) -> tuple[str, str]:
    if any(z3.is_real(v) for v in variables):
        return "QF_LRA", "Real"
    if any(_has_or(h) for h in hard):
        return "ALL", "Int"
    return "QF_LIA", "Int"


def _hard_to_smt2(hard, obj, sense: Sense) -> str:
    """把 ``(hard, obj, sense)`` 编成 z3 二进制可读的 OMT SMT-LIB2。"""
    variables = _collect_vars(hard, obj)
    logic, var_sort = _smt_logic_and_sort(hard, variables)
    obj_sexpr = obj.sexpr()
    sense_cmd = "maximize" if sense is Sense.MAX else "minimize"
    lines = [
        f"(set-logic {logic})",
        "(set-option :produce-models true)",
    ]
    for v in variables:
        lines.append(f"(declare-fun {v.decl().name()} () {var_sort})")
    for h in hard:
        lines.append(f"(assert {h.sexpr()})")
    lines.append(f"({sense_cmd} {obj_sexpr})")
    lines.append("(check-sat)")
    lines.append(f"(get-value ({obj_sexpr}))")
    return "\n".join(lines) + "\n"


def _parse_smt_num(token: str) -> Optional[Fraction]:
    token = token.strip()
    if not token:
        return None
    if token.startswith("(/"):
        inner = token.strip("()")
        parts = inner.split()
        if len(parts) == 3 and parts[0] == "/":
            return Fraction(int(parts[1]), int(parts[2]))
        return None
    try:
        return Fraction(int(token))
    except ValueError:
        return None


def _parse_get_value(stdout: str) -> Optional[Fraction]:
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("((("):
            continue
        m = re.search(r"\)\s+(\S+)\)\)$", line)
        if m is None:
            continue
        return _parse_smt_num(m.group(1))
    return None


def _parse_rlimit(stdout: str) -> int:
    m = re.search(r":rlimit-count\s+(\d+)", stdout)
    return int(m.group(1)) if m else 0


def _parse_sat(stdout: str) -> str:
    for line in stdout.splitlines():
        s = line.strip()
        if s in ("sat", "unsat", "unknown"):
            return s
    return "error"


def solve_binary(
    hard,
    obj,
    sense: Sense,
    *,
    z3_path: str | None = None,
    timeout_s: int = 120,
) -> dict:
    """用 z3 二进制（``z3 -st``）求解 OMT，返回 ``value``/``rlimit``（与 :func:`solve_native` 对齐）。"""
    exe = z3_path or shutil.which("z3")
    if not exe:
        raise FileNotFoundError("未找到 z3 二进制，请安装 z3 或通过 z3_path 指定")

    smt2 = _hard_to_smt2(hard, obj, sense)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".smt2", delete=False, encoding="ascii"
    ) as tmp:
        tmp.write(smt2)
        path = tmp.name
    try:
        t0 = perf_counter()
        proc = subprocess.run(
            [exe, "-st", path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        elapsed_ms = (perf_counter() - t0) * 1000.0
    except subprocess.TimeoutExpired:
        return {
            "value": None,
            "rlimit": None,
            "time_ms": timeout_s * 1000.0,
            "status": "timeout",
            "stderr": "timeout",
        }
    finally:
        Path(path).unlink(missing_ok=True)

    stdout = proc.stdout or ""
    status = _parse_sat(stdout)
    rlimit = _parse_rlimit(stdout)
    value = _parse_get_value(stdout) if status == "sat" else None
    return {
        "value": value,
        "rlimit": rlimit,
        "time_ms": elapsed_ms,
        "status": status,
        "stderr": (proc.stderr or "").strip(),
    }


__all__ = [
    "solve_omt_with_decider",
    "solve_native",
    "solve_binary",
]
