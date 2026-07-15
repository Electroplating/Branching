"""对比同一 OMT 实例在 z3 二进制与 ``solve_native``（Python API）上的用时与 rlimit。

对每条实例：
1. 导出 SMT-LIB2（``maximize``/``minimize`` + ``check-sat`` + ``get-value``）；
2. 调用 ``z3 -st`` 读取 ``:rlimit-count`` 与最优值；
3. 调用 :func:`omt_branching.solver.solve_native` 读取 API 侧 rlimit 与最优值；
4. 汇总 wall-clock 与 rlimit 比值。

运行::

    python -m examples.compare_native_vs_z3bin
    python -m examples.compare_native_vs_z3bin --test 10 --z3-path C:/path/to/z3.exe
    python -m examples.compare_native_vs_z3bin --demo
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Optional

import z3

from omt_branching.solver import (
    OMTInstance,
    generate_bool_lia_dataset,
    generate_instance,
    solve_native,
)
from omt_branching.solver.interfaces import Sense


def _sense_cmd(sense: Sense) -> str:
    return "maximize" if sense is Sense.MAX else "minimize"


def _var_sort(inst: OMTInstance) -> str:
    return "Real" if inst.theory == "LRA" else "Int"


def _set_logic(inst: OMTInstance) -> str:
    if inst.theory == "LRA":
        return "QF_LRA"
    if inst.family == "bool":
        return "ALL"
    return "QF_LIA"


def instance_to_smt2(inst: OMTInstance) -> str:
    """把 ``OMTInstance`` 编成 z3 二进制可读的 OMT SMT-LIB2。"""
    hard, obj, sense = inst.as_tuple()
    obj_sexpr = obj.sexpr()
    lines = [
        f"(set-logic {_set_logic(inst)})",
        "(set-option :produce-models true)",
    ]
    for v in inst.variables:
        lines.append(f"(declare-fun {v.decl().name()} () {_var_sort(inst)})")
    for h in hard:
        lines.append(f"(assert {h.sexpr()})")
    lines.append(f"({_sense_cmd(sense)} {obj_sexpr})")
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
    """从 ``(get-value ...)`` 输出解析目标最优值。"""
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("((("):
            continue
        # 形如 (((expr) value)) 或 (((expr) (/ n m)))
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


def solve_z3_binary(
    smt2: str,
    *,
    z3_path: str,
    timeout_s: int = 120,
) -> dict:
    """用 z3 二进制求解 SMT2，返回用时、rlimit 与最优值。"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".smt2", delete=False, encoding="ascii"
    ) as tmp:
        tmp.write(smt2)
        path = tmp.name
    try:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [z3_path, "-st", path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "value": None,
            "rlimit": None,
            "time_ms": timeout_s * 1000.0,
            "stderr": "timeout",
        }
    finally:
        Path(path).unlink(missing_ok=True)

    stdout = proc.stdout or ""
    return {
        "status": _parse_sat(stdout),
        "value": _parse_get_value(stdout),
        "rlimit": _parse_rlimit(stdout),
        "time_ms": elapsed_ms,
        "stderr": (proc.stderr or "").strip(),
    }


def solve_native_timed(
    hard,
    obj,
    sense: Sense,
    *,
    max_rlimit: int = -1,
) -> dict:
    """包装 :func:`solve_native` 并附加 wall-clock。"""
    t0 = time.perf_counter()
    out = solve_native(hard, obj, sense, max_rlimit=max_rlimit)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    status = "sat" if out.get("value") is not None else "unsat"
    return {
        "status": status,
        "value": out.get("value"),
        "rlimit": out.get("rlimit"),
        "time_ms": elapsed_ms,
    }


def build_demo_instance() -> OMTInstance:
    """与 ``examples/z3_demo.py`` 相同的有界 LIA 实例。"""
    x, y, z = z3.Int("x"), z3.Int("y"), z3.Int("z")
    hard = [
        x >= 0,
        x <= 12,
        y >= 0,
        y <= 12,
        z >= 0,
        z <= 12,
        x + y + z <= 18,
        2 * x + y <= 20,
        y + 3 * z <= 24,
    ]
    objective = 3 * x + 2 * y + 4 * z
    return OMTInstance(
        instance_id="demo_lia",
        variables=[x, y, z],
        hard=hard,
        objective=objective,
        sense=Sense.MAX,
        theory="LIA",
        family="box",
        description="z3_demo 有界 LIA，max 3x+2y+4z",
    )


def _fmt_value(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, Fraction):
        return str(v.numerator) if v.denominator == 1 else str(v)
    return str(v)


def _values_match(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return Fraction(a) == Fraction(b)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="z3 二进制 vs solve_native：用时与 rlimit 对比"
    )
    ap.add_argument("--z3-path", default=None, help="z3 可执行文件路径（默认同 PATH）")
    ap.add_argument("--test", type=int, default=5, help="随机实例数量（--demo 时忽略）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-vars", type=int, default=4)
    ap.add_argument("--max-vars", type=int, default=5)
    ap.add_argument(
        "--theory",
        choices=("lia", "bool_lia"),
        default="bool_lia",
        help="实例类型：纯 LIA 盒约束 或 带布尔结构的 bool LIA",
    )
    ap.add_argument("--demo", action="store_true", help="仅跑 z3_demo 单实例")
    ap.add_argument("--timeout", type=int, default=120, help="单实例 z3 二进制超时（秒）")
    ap.add_argument("--json", default=None, help="将逐实例结果写入 JSON 文件")
    ap.add_argument("--save-smt2", default=None, help="保存首条实例 SMT2 到指定路径")
    args = ap.parse_args()

    z3_path = args.z3_path or shutil.which("z3")
    if not z3_path:
        raise SystemExit("未找到 z3 二进制，请用 --z3-path 指定")

    if args.demo:
        instances = [build_demo_instance()]
    elif args.theory == "lia":
        rng = random.Random(args.seed)
        instances = [
            generate_instance(
                f"inst{i}",
                rng,
                n_vars=rng.randint(args.min_vars, args.max_vars),
            )
            for i in range(args.test)
        ]
    else:
        instances = generate_bool_lia_dataset(
            args.test, seed=args.seed, min_vars=args.min_vars, max_vars=args.max_vars
        )

    if args.save_smt2 and instances:
        Path(args.save_smt2).write_text(
            instance_to_smt2(instances[0]), encoding="ascii"
        )

    rows: list[dict] = []
    for inst in instances:
        smt2 = instance_to_smt2(inst)
        hard, obj, sense = inst.as_tuple()
        binary = solve_z3_binary(smt2, z3_path=z3_path, timeout_s=args.timeout)
        native = solve_native_timed(hard, obj, sense)
        match = _values_match(native["value"], binary["value"])
        rlimit_ratio = None
        if native["rlimit"] and binary["rlimit"]:
            rlimit_ratio = native["rlimit"] / binary["rlimit"]
        time_ratio = None
        if native["time_ms"] and binary["time_ms"]:
            time_ratio = native["time_ms"] / binary["time_ms"]
        row = {
            "id": inst.instance_id,
            "theory": inst.theory,
            "family": inst.family,
            "native": native,
            "binary": binary,
            "value_match": match,
            "rlimit_ratio_native_over_binary": rlimit_ratio,
            "time_ratio_native_over_binary": time_ratio,
        }
        rows.append(row)

    ok = [r for r in rows if r["native"]["status"] == "sat" and r["binary"]["status"] == "sat"]
    n = max(1, len(ok))

    def _avg(key: str, side: str) -> float:
        vals = [r[side][key] for r in ok if r[side].get(key) is not None]
        return sum(vals) / max(1, len(vals))

    print(f"=== z3 二进制 ({z3_path}) vs solve_native ===")
    print(f"实例: {len(instances)}，双方 SAT: {len(ok)}，最优值一致: {sum(r['value_match'] for r in ok)}/{len(ok)}")
    print()
    print(
        f"{'id':<12} {'native_rl':>10} {'bin_rl':>10} {'rl_ratio':>9} "
        f"{'native_ms':>10} {'bin_ms':>10} {'t_ratio':>9} {'match':>6}"
    )
    print("-" * 78)
    for r in rows:
        nat, bin_ = r["native"], r["binary"]
        rl_r = (
            f"{r['rlimit_ratio_native_over_binary']:.2f}"
            if r["rlimit_ratio_native_over_binary"] is not None
            else "—"
        )
        t_r = (
            f"{r['time_ratio_native_over_binary']:.2f}"
            if r["time_ratio_native_over_binary"] is not None
            else "—"
        )
        print(
            f"{r['id']:<12} "
            f"{nat.get('rlimit') or 0:>10} "
            f"{bin_.get('rlimit') or 0:>10} "
            f"{rl_r:>9} "
            f"{nat.get('time_ms', 0):>10.1f} "
            f"{bin_.get('time_ms', 0):>10.1f} "
            f"{t_r:>9} "
            f"{('Y' if r['value_match'] else 'N'):>6}"
        )
        if not r["value_match"]:
            print(
                f"  !! 最优值不一致: native={_fmt_value(nat.get('value'))} "
                f"binary={_fmt_value(bin_.get('value'))}"
            )
        if bin_.get("stderr"):
            print(f"  binary stderr: {bin_['stderr'][:120]}")

    print()
    print("--- 汇总（仅双方 SAT 实例）---")
    print(f"  平均 rlimit  native={_avg('rlimit', 'native'):.0f}  binary={_avg('rlimit', 'binary'):.0f}")
    print(f"  平均用时(ms) native={_avg('time_ms', 'native'):.1f}  binary={_avg('time_ms', 'binary'):.1f}")
    ratios = [r["rlimit_ratio_native_over_binary"] for r in ok if r["rlimit_ratio_native_over_binary"]]
    t_ratios = [r["time_ratio_native_over_binary"] for r in ok if r["time_ratio_native_over_binary"]]
    if ratios:
        print(f"  平均 rlimit 比(native/binary) = {sum(ratios)/len(ratios):.2f}")
    if t_ratios:
        print(f"  平均用时比(native/binary)   = {sum(t_ratios)/len(t_ratios):.2f}")

    if args.json:
        payload = {
            "z3_path": z3_path,
            "count": len(instances),
            "sat_both": len(ok),
            "rows": rows,
        }
        Path(args.json).write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        print(f"\n结果已写入 {args.json}")


if __name__ == "__main__":
    main()
