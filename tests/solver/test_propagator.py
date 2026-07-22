from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")

from omt_branching.solver.propagator import LearnedDecidePropagator
from omt_branching.solver.propagator_snapshot import atom_key


def _sat_instance():
    xs = [z3.Bool(f"b{i}") for i in range(12)]
    clauses = [z3.Or(xs[i], z3.Not(xs[(i + 1) % 12]), xs[(i + 2) % 12]) for i in range(12)]
    return xs, clauses


def _solve(decider):
    xs, clauses = _sat_instance()
    s = z3.Solver()
    p = LearnedDecidePropagator(s, xs, decider)
    s.add(*clauses)
    return s.check(), p.n_decisions


def test_propagator_controls_decisions_and_preserves_correctness():
    idx = lambda k: int(k[1:])
    resA, nA = _solve(lambda und, asg, trail=None: (min(und, key=idx), True))
    resB, nB = _solve(lambda und, asg, trail=None: (max(und, key=idx), True))
    assert resA == resB == z3.sat        # 正确性不变
    assert nA > 0 and nB > 0             # 两个 decider 都真的强制了决策


def test_none_decider_falls_back():
    resN, nN = _solve(lambda und, asg, trail=None: None)   # 永远 None = 退回 VSIDS
    assert resN == z3.sat
    assert nN == 0                        # 我们没强制任何决策


def test_decide_counters_defer_vs_next_split():
    """on_decide = next_split + defer + empty + bad_key；故 on_decide - next_split ≠ defer。"""
    xs, clauses = _sat_instance()
    s = z3.Solver()
    p = LearnedDecidePropagator(s, xs, lambda und, asg, trail=None: None)
    s.add(*clauses)
    assert s.check() == z3.sat
    assert p.n_next_split == 0
    assert p.n_on_decide == (
        p.n_next_split + p.n_defer + p._counters.empty + p._counters.bad_key
    )
    assert p.n_on_decide - p.n_next_split == p.n_defer + p._counters.empty + p._counters.bad_key
    # 求解结束：分支原子应均已定，未定集为空
    assert p._undecided == set()
    assert len(p._val) == len(p.key2atom)


def test_branch_action_logits_log_n_when_sample():
    """sample=True 时 defer += log n，等 logit 下 P(defer)=1/2。"""
    import math
    import torch
    from omt_branching.solver.rl_decide import _branch_action_logits

    defer = torch.zeros(())
    atoms = torch.zeros(4)
    logits = _branch_action_logits(defer, atoms, sample=True)
    assert float(logits[0]) == pytest.approx(math.log(4))
    probs = torch.softmax(logits, dim=0)
    assert float(probs[0]) == pytest.approx(0.5, abs=1e-5)
    assert float(probs[1]) == pytest.approx(0.125, abs=1e-5)
    logits_arg = _branch_action_logits(defer, atoms, sample=False)
    assert float(logits_arg[0]) == pytest.approx(0.0)


def test_pop_notifies_decider_on_backtrack():
    """propagator.pop 会调用 decider.on_backtrack(num_scopes)。"""
    xs, clauses = _sat_instance()
    events = []

    class _Dec:
        def on_backtrack(self, num_scopes=1):
            events.append(num_scopes)

        def __call__(self, undecided, assignment, trail=None):
            return None

    s = z3.Solver()
    p = LearnedDecidePropagator(s, xs, _Dec())
    s.add(*clauses)
    p.push()
    p.push()
    # 模拟注册原子被 fixed，再 pop 两层
    k0 = atom_key(xs[0])
    p._val[k0] = True
    p._trail.append(k0)
    p.pop(2)
    assert events == [2]
    assert k0 not in p._val


def test_watch_all_branch_subset_undecided():
    """全量 watch 收 fixed；仅 branch 进入 undecided。"""
    x = z3.Int("x")
    a, b = x >= 5, x <= 2
    box = x >= 0
    s = z3.Solver()
    watch = [a, b, box]
    branch = [a, b]
    seen = []

    def decider(und, asg, trail=None):
        seen.append(list(und))
        return None

    p = LearnedDecidePropagator(s, watch, decider, branch_atoms=branch)
    assert p.branch_keys == {atom_key(a), atom_key(b)}
    assert atom_key(box) in p.key2atom
    assert atom_key(box) not in p._undecided
    s.add(z3.Or(a, b), box)
    assert s.check() == z3.sat
    # decide 候选不应包含 box
    for und in seen:
        assert atom_key(box) not in und

