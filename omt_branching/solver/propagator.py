"""LearnedDecidePropagator：经 z3 UserPropagator 的 add_decide/next_split **接管 z3 内部
布尔分支决策**（不改 z3）。decider 不自信（返回 None）时直接放行 -> 退回 z3 原生 VSIDS。

注意：基类占用属性名 ``fixed``/``decide``，本类用 ``_val``/``_on_decide`` 等避让。

原子分两集：

- ``atoms``（watch）：图上全量布尔原子，全部 ``prop.add``，以收齐 ``fixed`` 赋值；
- ``branch_atoms``：原析取子句注册集，才进入 ``_undecided`` 参与分支选择。

未定集 ``_undecided`` 在 ``_on_fixed`` / ``pop`` 增量维护（仅 branch 键），避免每次
decide 全表扫描。赋值 ``_trail`` 保序，随 ``_val`` 一并传给 decider。

计数（``fresh`` 副本共享同一对象，便于汇总）::

- ``n_on_decide``：``_on_decide`` 回调次数；
- ``n_next_split`` / ``n_decisions``：实际 ``next_split`` 次数（GNN override）；
- ``n_defer``：decider 返回 ``None``、放行 VSIDS 的次数。

``n_on_decide - n_next_split`` **不等于** ``n_defer``：前者还含「无未定原子」
与「非法 key」等早退，故 ``n_defer`` 单独累计。
"""
from __future__ import annotations

from typing import Callable, Optional

import z3

from omt_branching.solver.propagator_snapshot import atom_key


class _DecideCounters:
    """可变计数器；``fresh()`` 后各 propagator 实例共享同一对象。"""

    __slots__ = ("on_decide", "next_split", "defer", "empty", "bad_key")

    def __init__(self) -> None:
        self.on_decide = 0
        self.next_split = 0
        self.defer = 0
        self.empty = 0
        self.bad_key = 0


class LearnedDecidePropagator(z3.UserPropagateBase):
    def __init__(
        self,
        s,
        atoms,
        decider: Callable,
        *,
        branch_atoms=None,
        _counters: Optional[_DecideCounters] = None,
    ):
        super().__init__(s)
        # watch：全量 add，收 fixed
        self.atoms = list(atoms)
        self.key2atom = {atom_key(a): a for a in self.atoms}
        # z3 每次回调都新建 t 的 Python 包装（id() 不稳定），但底层 AST 的 get_id() 稳定。
        # 注册原子被 self.atoms 钉住存活整个求解，其 get_id() 不会被回收复用，故可安全建表：
        # _on_fixed 里用 get_id() O(1) 命中，避免每次回调对原子做 str()（实测占总耗时 ~65%）。
        self._id2key = {a.get_id(): k for k, a in self.key2atom.items()}
        # branch：仅这些键进 undecided / 允许 next_split
        if branch_atoms is None:
            branch_list = list(self.atoms)
        else:
            branch_list = list(branch_atoms)
        self.branch_atoms = branch_list
        self.branch_keys = {atom_key(a) for a in branch_list}
        # 分支原子必须也能 watch（否则无法 next_split）
        for a in branch_list:
            k = atom_key(a)
            if k not in self.key2atom:
                self.atoms.append(a)
                self.key2atom[k] = a
                self._id2key[a.get_id()] = k
        self.decider = decider
        self._val: dict = {}          # key -> bool（当前赋值，插入序≈trail）
        self._trail: list = []        # 赋值先后（保序）
        self._lim: list = []
        # 仅分支原子未定集：fixed 时剔除，pop 时加回
        self._undecided: set[str] = set(self.branch_keys)
        self._counters = _counters if _counters is not None else _DecideCounters()
        self.add_fixed(self._on_fixed)
        self.add_decide(self._on_decide)
        for a in self.atoms:
            self.add(a)               # 全量注册，z3 才回调 fixed / decide

    @property
    def n_on_decide(self) -> int:
        return self._counters.on_decide

    @property
    def n_next_split(self) -> int:
        return self._counters.next_split

    @property
    def n_defer(self) -> int:
        return self._counters.defer

    @property
    def n_decisions(self) -> int:
        """兼容旧名：等同 ``n_next_split``。"""
        return self._counters.next_split

    @n_decisions.setter
    def n_decisions(self, value: int) -> None:
        self._counters.next_split = int(value)

    def push(self):
        self._lim.append(len(self._trail))

    def pop(self, num_scopes):
        for _ in range(num_scopes):
            lim = self._lim.pop()
            while len(self._trail) > lim:
                k = self._trail.pop()
                self._val.pop(k, None)
                if k in self.branch_keys:
                    self._undecided.add(k)
        # 冲突回退（及任意 scope pop）后通知 decider：下次 decide 立刻 refocus。
        on_bt = getattr(self.decider, "on_backtrack", None)
        if callable(on_bt):
            on_bt(num_scopes)

    def fresh(self, new_ctx):
        return LearnedDecidePropagator(
            new_ctx,
            self.atoms,
            self.decider,
            branch_atoms=self.branch_atoms,
            _counters=self._counters,
        )

    def _on_fixed(self, t, v):
        # get_id() 命中已 watch 原子（z3 只对 add 过的项回调 fixed），避免 str(t)。
        k = self._id2key.get(t.get_id())
        if k is not None and k not in self._val:
            self._val[k] = z3.is_true(v)
            self._trail.append(k)
            if k in self.branch_keys:
                self._undecided.discard(k)

    def _ordered_assignment(self) -> dict:
        """按 trail 保序的赋值视图（供 decider / 建图）。"""
        return {k: self._val[k] for k in self._trail}

    def _on_decide(self, t, idx, phase):
        self._counters.on_decide += 1
        if not self._undecided:
            self._counters.empty += 1
            return
        # 稳定顺序，便于与采样 locs 对齐；赋值按 trail 保序传给 decider。
        undecided = sorted(self._undecided)
        choice = self.decider(undecided, self._ordered_assignment(), self._trail)
        if choice is None:
            self._counters.defer += 1
            return                    # 退回 VSIDS
        key, ph = choice
        atom = self.key2atom.get(key)
        if atom is None or key not in self.branch_keys:
            self._counters.bad_key += 1
            return
        self._counters.next_split += 1
        # z3 next_split 的 phase 是 Z3_lbool：真=1，假=-1，未定=0（≠ Python bool）
        z3_phase = z3.Z3_L_TRUE if ph else z3.Z3_L_FALSE
        self.next_split(atom, 0, z3_phase)


__all__ = ["LearnedDecidePropagator"]
