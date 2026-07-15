"""``SolveBackend`` 的 z3 实现（adapter 层，集中所有 z3 依赖）。

通过 z3 的公开 Python API 提供 GOMT calculus 所需的 ``Solve`` / ``Optimize`` 与
公式代数。**不修改、不依赖系统 z3 二进制**，仅用 pip 安装的 ``z3-solver`` wheel
（与系统 ``/usr/local/bin/z3`` 版本一致，互不影响）。

每次 ``Solver`` / ``Optimize`` 在**独立** ``z3.Context`` 中运行，使 ``rlimit count``
自该次会话从 0 起计，不受进程内默认 context 累积污染。外部公式仍在默认 context
构造；进入求解前 ``translate`` 迁入，返回的 model 再 ``translate`` 回默认 context。
"""

from __future__ import annotations

from fractions import Fraction
from typing import Optional

import z3

from omt_branching.solver.interfaces import Sense


class Unbounded(Exception):
    """目标在某 branch 上无界（v1 不处理无界优化）。"""


def _translate(expr, ctx: z3.Context):
    """把表达式复制到 ``ctx``；已在目标 context 则原样返回。"""
    if expr is None:
        return None
    if expr.ctx == ctx:
        return expr
    return expr.translate(ctx)


def _translate_model(model, ctx: z3.Context):
    """把 model 复制到 ``ctx``，便于与默认 context 中的 ``term`` 一起 ``eval``。"""
    if model is None:
        return None
    if model.ctx == ctx:
        return model
    return model.translate(ctx)


class Z3Backend:
    """以 z3 公开 API 实现的 ``SolveBackend``。

    额外累计 z3 的 **rlimit count**（``Solver.statistics()`` 的 ``"rlimit count"``）
    与 solve 次数：rlimit 是 z3 与硬件/负载无关的确定性工作量计量，用它的增长反映
    “求解耗时”比 wall-clock 更稳定可复现，适合作强化学习的代价信号。
    """

    def __init__(self, eps: float = 1e-9):
        self.eps = eps
        self.rlimit_count = 0    # 累计 rlimit（跨本 backend 的所有 solve/optimize 调用）
        self.solve_calls = 0     # 累计 check() 次数
        # 增量求解：持久的 solver/optimizer（base=φ 断言一次，branch=ψ 经 push/pop 逐节点变化）。
        self._inc_solver = None
        self._inc_solver_ctx: Optional[z3.Context] = None
        self._inc_solver_base = None
        self._inc_solver_rl = 0          # 该持久 solver 上次见到的累计 rlimit（用于取增量）
        self._inc_opt = None
        self._inc_opt_ctx: Optional[z3.Context] = None
        self._inc_opt_base = None
        self._inc_opt_obj = None
        self._inc_opt_sense = None
        self._inc_opt_rl = 0

    def reset_stats(self) -> None:
        """清零累计统计（复用同一 backend 跑多次时使用）。"""
        self.rlimit_count = 0
        self.solve_calls = 0
        self._inc_solver = None
        self._inc_solver_ctx = None
        self._inc_solver_base = None
        self._inc_solver_rl = 0
        self._inc_opt = None
        self._inc_opt_ctx = None
        self._inc_opt_base = None
        self._inc_opt_obj = None
        self._inc_opt_sense = None
        self._inc_opt_rl = 0

    # ---------------- 求解（一次性：native / 初始态）----------------
    def solve(self, constraint) -> Optional[z3.ModelRef]:
        ctx = z3.Context()
        s = z3.Solver(ctx=ctx)
        s.add(_translate(constraint, ctx))
        res = s.check()
        self.solve_calls += 1
        self.rlimit_count += self._rlimit(s)
        if res != z3.sat:
            return None
        return _translate_model(s.model(), constraint.ctx)

    def optimize(self, constraint, objective, sense: Sense):
        ctx = z3.Context()
        o = z3.Optimize(ctx=ctx)
        obj = _translate(objective, ctx)
        o.add(_translate(constraint, ctx))
        if sense is Sense.MIN:
            o.minimize(obj)
        else:
            o.maximize(obj)
        res = o.check()
        self.solve_calls += 1
        self.rlimit_count += self._rlimit(o)
        if res != z3.sat:
            return None
        m = o.model()
        # 用最优 model 上的目标取值作为最优值（LIA/LRA 闭最优时即为 bound，
        # 且避免 lower/upper 返回 epsilon/oo 表达式带来的解析问题）。
        val = self._num(m.eval(obj, model_completion=True))
        return _translate_model(m, objective.ctx), val

    # ---------------- 增量求解（GOMT 热回路 / strong-branching 标签）----------------
    def solve_branch(self, base, branch) -> Optional[z3.ModelRef]:
        """增量 ``Solve(base∧branch)``：``base`` 固定断言一次，``branch`` 经 push/pop 变化。

        复用持久 solver 保留 z3 已学到的 lemma，避免每个 GOMT 节点从零重解 φ。
        z3 模型取得后即使 pop / 再 check 仍有效（快照语义），可安全存作 incumbent。
        """
        if self._inc_solver is None or self._inc_solver_base is not base:
            ctx = z3.Context()
            s = z3.Solver(ctx=ctx)
            s.add(_translate(base, ctx))
            self._inc_solver = s
            self._inc_solver_ctx = ctx
            self._inc_solver_base = base
            self._inc_solver_rl = 0
        s = self._inc_solver
        ctx = self._inc_solver_ctx
        s.push()
        s.add(_translate(branch, ctx))
        res = s.check()
        model = _translate_model(s.model(), base.ctx) if res == z3.sat else None
        s.pop()
        self.solve_calls += 1
        self._inc_solver_rl = self._accumulate_delta(s, self._inc_solver_rl)
        return model

    def optimize_branch(self, base, branch, objective, sense: Sense):
        """增量 ``Optimize(base∧branch)``：``base``+目标断言一次，``branch`` 经 push/pop 变化。"""
        if (self._inc_opt is None or self._inc_opt_base is not base
                or self._inc_opt_obj is not objective or self._inc_opt_sense is not sense):
            ctx = z3.Context()
            o = z3.Optimize(ctx=ctx)
            obj = _translate(objective, ctx)
            o.add(_translate(base, ctx))
            if sense is Sense.MIN:
                o.minimize(obj)
            else:
                o.maximize(obj)
            self._inc_opt = o
            self._inc_opt_ctx = ctx
            self._inc_opt_base = base
            self._inc_opt_obj = objective
            self._inc_opt_sense = sense
            self._inc_opt_rl = 0
        o = self._inc_opt
        ctx = self._inc_opt_ctx
        o.push()
        o.add(_translate(branch, ctx))
        res = o.check()
        if res != z3.sat:
            o.pop()
            self.solve_calls += 1
            self._inc_opt_rl = self._accumulate_delta(o, self._inc_opt_rl)
            return None
        m = o.model()
        obj = _translate(objective, ctx)
        val = self._num(m.eval(obj, model_completion=True))
        o.pop()
        self.solve_calls += 1
        self._inc_opt_rl = self._accumulate_delta(o, self._inc_opt_rl)
        return _translate_model(m, objective.ctx), val

    def _rlimit(self, solver) -> int:
        """读取该 solver 当前累计 rlimit count（缺失返回 0）。"""
        try:
            st = solver.statistics()
            for key in st.keys():
                if key == "rlimit count":
                    return int(st.get_key_value(key))
        except Exception:  # pragma: no cover - 统计缺失不应影响求解
            pass
        return 0

    def _accumulate_delta(self, solver, prev: int) -> int:
        """持久 solver 的 rlimit 是**累计值**，只把本次增量计入 backend 统计；返回新累计值。"""
        cur = self._rlimit(solver)
        self.rlimit_count += max(0, cur - prev)
        return cur

    # ---------------- 取值 ----------------
    def value(self, model, term):
        model = _translate_model(model, term.ctx)
        return self._num(model.eval(term, model_completion=True))

    def is_true(self, model, atom) -> bool:
        model = _translate_model(model, atom.ctx)
        return z3.is_true(model.eval(atom, model_completion=True))

    # ---------------- 公式代数 ----------------
    def conjoin(self, *constraints):
        if not constraints:
            return z3.BoolVal(True)
        if len(constraints) == 1:
            return constraints[0]
        return z3.And(*constraints)

    def negate(self, constraint):
        return z3.Not(constraint)

    def better(self, objective, value, sense: Sense):
        num = self._mk_numeral(objective, value)
        return objective < num if sense is Sense.MIN else objective > num

    def top(self):
        return z3.BoolVal(True)

    def le(self, term, bound):
        return term <= self._coerce_bound(term, bound)

    def ge(self, term, bound):
        return term >= self._coerce_bound(term, bound)

    def _coerce_bound(self, term, bound):
        """把 python int/Fraction 的界转成与 ``term`` sort 匹配的 z3 常量。"""
        if isinstance(bound, (int, Fraction)):
            return self._mk_numeral(term, bound)
        return bound

    # ---------------- 内部工具 ----------------
    def _num(self, ref):
        """z3 数值 ref -> python ``int`` / ``Fraction``；无穷/epsilon 抛 ``Unbounded``。"""
        if z3.is_int_value(ref):
            return ref.as_long()
        if z3.is_rational_value(ref):
            return Fraction(ref.numerator_as_long(), ref.denominator_as_long())
        text = str(ref)
        if "oo" in text or "epsilon" in text:
            raise Unbounded(text)
        try:
            return Fraction(text)
        except (ValueError, ZeroDivisionError) as exc:  # pragma: no cover
            raise Unbounded(text) from exc

    def _mk_numeral(self, term, value):
        """按 ``term`` 的 sort 构造与 ``value`` 对应的 z3 数值常量。"""
        if isinstance(value, Fraction) and value.denominator == 1:
            value = value.numerator
        if isinstance(value, int):
            return z3.IntVal(value) if z3.is_int(term) else z3.RealVal(value)
        # Fraction（实数）
        return z3.RealVal(f"{value.numerator}/{value.denominator}")


__all__ = ["Z3Backend", "Unbounded"]
