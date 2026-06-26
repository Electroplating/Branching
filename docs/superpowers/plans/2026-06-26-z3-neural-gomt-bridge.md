# z3 ↔ Neural GOMT Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean, non-invasive API bridge that lets the existing Neural branching policy drive a z3-backed OMT search implemented as the GOMT calculus.

**Architecture:** A theory-agnostic GOMT calculus engine (`GOMTState ⟨I,Δ,τ⟩`, rules F-Split/F-Sat/F-Close) consults a `BranchingStrategy` for the one heuristic freedom (F-Split) and delegates `Solve`/`Optimize` to a `SolveBackend`. z3 is confined to two adapter modules (`z3_backend.py`, `extractor.py`). The Neural policy plugs in via three seams: z3-state→`SolverSnapshot` (extractor), `advise` (existing), `BranchingAdvice`→split formulas.

**Tech Stack:** Python 3.10 (conda `omt`), z3-solver 4.15.4.0, PyTorch 2.7.1 (existing Neural side), pytest (dev).

## Global Constraints

- Python 3.10; run everything via `conda run -n omt ...`.
- Runtime dep added: `z3-solver==4.15.4.0` (matches system `/usr/local/bin/z3` 4.15.4; separate wheel — never touch system z3). Dev dep: `pytest`.
- Docs/comments in **Chinese**; `__all__`, identifiers, type annotations in English (AGENTS.md §6).
- Every module starts with `from __future__ import annotations`.
- **z3 import is confined to `omt_branching/solver/z3_backend.py` and `omt_branching/solver/extractor.py`.** `interfaces.py`, `problem.py`, `calculus.py` MUST NOT import z3.
- Do NOT modify Neural side (`input/`, `model/`, `output/`) behavior. Reuse `SolverSnapshot` / `BranchingAdvice` / `BranchingPolicyService` unchanged.
- Config = frozen dataclass. No PyG/new heavy deps.
- v1 scope: single-objective **minimize/maximize** over **LIA/LRA**, bounded objective. Multi-objective is an extension point only.
- Tests live in `Branching/tests/solver/`; run `conda run -n omt python -m pytest tests/solver -v` from `Branching/`.

**Opaque handle types** (aliases in `interfaces.py`, all `= Any`): `Model`, `Term`, `Constraint`, `Atom`, `Value`. The calculus and strategies treat these as opaque and manipulate them ONLY through `SolveBackend`.

---

### Task 0: Package + test scaffold

**Files:**
- Create: `omt_branching/solver/__init__.py` (empty for now, exports added in Task 8)
- Create: `tests/solver/__init__.py` (empty)
- Create: `tests/solver/conftest.py`

**Interfaces:**
- Produces: a `random_lia_instance(seed)` pytest helper used by later oracle tests.

- [ ] **Step 1: Create empty package files**

`omt_branching/solver/__init__.py`:
```python
"""z3 ↔ Neural GOMT 桥接子包。公共导出见 Task 8。"""
```
`tests/solver/__init__.py`: empty file.

- [ ] **Step 2: Write conftest with z3 availability guard + instance generator**

`tests/solver/conftest.py`:
```python
from __future__ import annotations
import pytest

z3 = pytest.importorskip("z3")  # 整个 solver 测试集需要 z3


def random_lia_instance(seed: int, n_vars: int = 4, n_constraints: int = 6,
                        sense_min: bool = True):
    """构造一个有界的 LIA 单目标实例。

    返回 (hard_constraints: list[z3.BoolRef], variables: list[z3.ArithRef],
          objective: z3.ArithRef, sense_min: bool)。
    保证有界：每个变量加 0 <= x <= 20 的盒约束。
    """
    import random
    rng = random.Random(seed)
    xs = [z3.Int(f"x{i}") for i in range(n_vars)]
    hard = []
    for x in xs:
        hard.append(x >= 0)
        hard.append(x <= 20)
    for _ in range(n_constraints):
        coeffs = [rng.randint(-3, 3) for _ in xs]
        rhs = rng.randint(0, 40)
        expr = z3.Sum([c * x for c, x in zip(coeffs, xs)])
        hard.append(expr <= rhs)
    obj = z3.Sum([rng.randint(1, 4) * x for x in xs])
    return hard, xs, obj, sense_min


@pytest.fixture
def lia_instance():
    return random_lia_instance(seed=0)
```

- [ ] **Step 3: Verify scaffold imports**

Run: `conda run -n omt python -m pytest tests/solver -v`
Expected: `no tests ran` (0 collected), exit 0 — confirms collection + z3 import work.

- [ ] **Step 4: Stage (commit deferred per session policy)**

```bash
git add omt_branching/solver/__init__.py tests/solver/
```

---

### Task 1: Core interfaces & data types (`interfaces.py`)

**Files:**
- Create: `omt_branching/solver/interfaces.py`
- Test: `tests/solver/test_interfaces.py`

**Interfaces:**
- Produces:
  - `class Sense(enum.Enum): MIN; MAX`
  - Type aliases `Model = Term = Constraint = Atom = Value = Any`
  - `@dataclass class SplitDecision`: `kind: str` (`"split"` | `"resolve"`), `subformulas: list[Constraint] = []` (ordered; first is explored first), `info: dict = {}`. Helpers `SplitDecision.split(formulas)`, `SplitDecision.resolve()`.
  - `@dataclass class GOMTState`: `incumbent: Optional[Model]`, `delta: Constraint`, `tau: list[Constraint]`, `objective: Term`, `sense: Sense`, `hard: Constraint`, `step: int = 0`, `stats: dict = {}`. Property `top` → `tau[0]`; `saturated` → `not tau`.
  - `class SolveBackend(Protocol)`: `solve(constraint) -> Optional[Model]`; `optimize(constraint, objective, sense) -> Optional[tuple[Model, Value]]`; `value(model, term) -> Value`; `is_true(model, atom) -> bool`; `conjoin(*constraints) -> Constraint`; `negate(constraint) -> Constraint`; `better(objective, value, sense) -> Constraint`; `top() -> Constraint`; `le(term, bound) -> Atom`; `ge(term, bound) -> Atom`.
  - `class BranchingStrategy(Protocol)`: `propose(state: GOMTState, backend: SolveBackend) -> SplitDecision`.

- [ ] **Step 1: Write the failing test**

`tests/solver/test_interfaces.py`:
```python
from __future__ import annotations
from omt_branching.solver.interfaces import (
    Sense, SplitDecision, GOMTState, SolveBackend, BranchingStrategy,
)


def test_sense_members():
    assert {s.name for s in Sense} == {"MIN", "MAX"}


def test_split_decision_helpers():
    s = SplitDecision.split(["a", "b"])
    assert s.kind == "split" and s.subformulas == ["a", "b"]
    r = SplitDecision.resolve()
    assert r.kind == "resolve" and r.subformulas == []


def test_gomt_state_top_and_saturated():
    st = GOMTState(incumbent=None, delta="D", tau=["t0", "t1"],
                   objective="t", sense=Sense.MIN, hard="phi")
    assert st.top == "t0"
    assert st.saturated is False
    st.tau = []
    assert st.saturated is True


def test_protocols_are_runtime_checkable():
    class B:
        def solve(self, c): ...
        def optimize(self, c, o, s): ...
        def value(self, m, t): ...
        def is_true(self, m, a): ...
        def conjoin(self, *c): ...
        def negate(self, c): ...
        def better(self, o, v, s): ...
        def top(self): ...
        def le(self, t, b): ...
        def ge(self, t, b): ...
    assert isinstance(B(), SolveBackend)

    class S:
        def propose(self, state, backend): ...
    assert isinstance(S(), BranchingStrategy)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n omt python -m pytest tests/solver/test_interfaces.py -v`
Expected: FAIL — `ModuleNotFoundError: omt_branching.solver.interfaces`.

- [ ] **Step 3: Write `interfaces.py`**

Implement exactly the Produces list. Use `from typing import Any, Optional, Protocol, runtime_checkable`. Decorate both Protocols with `@runtime_checkable`. `SplitDecision`/`GOMTState` are `@dataclass`; use `field(default_factory=...)` for mutable defaults. Chinese docstrings on each class. `__all__` lists all 7 names.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n omt python -m pytest tests/solver/test_interfaces.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Stage**

```bash
git add omt_branching/solver/interfaces.py tests/solver/test_interfaces.py
```

---

### Task 2: z3 backend (`z3_backend.py`)

**Files:**
- Create: `omt_branching/solver/z3_backend.py`
- Test: `tests/solver/test_z3_backend.py`

**Interfaces:**
- Consumes: `Sense` from `interfaces`.
- Produces: `class Z3Backend(SolveBackend)` with the 10 methods. Construction: `Z3Backend(eps: float = 1e-9)`. Semantics:
  - `solve(c)`: `z3.Solver()`, add `c`, return `model` if `sat` else `None`.
  - `optimize(c, obj, sense)`: `z3.Optimize()`, add `c`, `h = o.minimize(obj)` or `o.maximize(obj)`; if `o.check()==sat` return `(o.model(), _num(o.lower(h)|o.upper(h)))` else `None`. If bound is `oo`/unbounded, raise `Unbounded`.
  - `value(m, t)`: `_num(m.eval(t, model_completion=True))` → `Fraction` (or `int`).
  - `is_true(m, atom)`: `z3.is_true(m.eval(atom, model_completion=True))`.
  - `conjoin(*cs)`: `z3.And(*cs)` (empty → `z3.BoolVal(True)`).
  - `negate(c)`: `z3.Not(c)`.
  - `better(obj, value, sense)`: `obj < value` (MIN) / `obj > value` (MAX). `value` may be `Fraction`/`int` → `z3.RealVal`/`IntVal` per `obj.sort()`.
  - `top()`: `z3.BoolVal(True)`.
  - `le(t, b)` / `ge(t, b)`: `t <= b` / `t >= b`.
  - Helper `_num(ref)`: `IntNumRef→int`; `RatNumRef→Fraction(num, den)`.

- [ ] **Step 1: Write the failing test**

`tests/solver/test_z3_backend.py`:
```python
from __future__ import annotations
from fractions import Fraction
import pytest
z3 = pytest.importorskip("z3")
from omt_branching.solver.interfaces import Sense
from omt_branching.solver.z3_backend import Z3Backend


def test_solve_sat_and_unsat():
    b = Z3Backend()
    x = z3.Int("x")
    assert b.solve(z3.And(x > 0, x < 3)) is not None
    assert b.solve(z3.And(x > 3, x < 1)) is None


def test_optimize_matches_native_min_max():
    b = Z3Backend()
    x, y = z3.Int("x"), z3.Int("y")
    hard = z3.And(x + y <= 10, x >= 0, y >= 0)
    m, v = b.optimize(hard, 2 * x + y, Sense.MAX)
    assert v == 20
    m2, v2 = b.optimize(hard, x + y, Sense.MIN)
    assert v2 == 0


def test_value_int_and_rational():
    b = Z3Backend()
    x = z3.Real("x")
    m = b.solve(2 * x == 1)
    assert b.value(m, x) == Fraction(1, 2)


def test_better_constraint_min():
    b = Z3Backend()
    x = z3.Int("x")
    # better-than-5 for MIN means x < 5; x==4 satisfies, x==6 does not
    c = b.better(x, 5, Sense.MIN)
    assert b.solve(z3.And(c, x == 4)) is not None
    assert b.solve(z3.And(c, x == 6)) is None


def test_le_ge_and_conjoin_negate_top():
    b = Z3Backend()
    x = z3.Int("x")
    assert b.solve(z3.And(b.le(x, 3), b.ge(x, 3), x == 3)) is not None
    assert b.solve(b.conjoin(b.le(x, 1), b.ge(x, 5))) is None
    assert b.solve(b.conjoin()) is not None          # empty conjunction = True
    assert b.solve(z3.And(b.negate(x == 0), x == 0)) is None
    assert b.solve(b.top()) is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n omt python -m pytest tests/solver/test_z3_backend.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `z3_backend.py`**

Import z3. Define `class Unbounded(Exception)`. Implement all 10 methods + `_num` per Produces. `better`: inspect `obj.sort()` — if `z3.is_int(obj)` and value is integral use `z3.IntVal`, else `z3.RealVal(str(value))` (Fraction → exact). For `optimize`, read `o.lower(h)` for MIN / `o.upper(h)` for MAX; guard `_num` against `oo`/`-oo` (str starts with check) → raise `Unbounded`. `__all__ = ["Z3Backend", "Unbounded"]`.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n omt python -m pytest tests/solver/test_z3_backend.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Stage**

```bash
git add omt_branching/solver/z3_backend.py tests/solver/test_z3_backend.py
```

---

### Task 3: GOMT problem (`problem.py`)

**Files:**
- Create: `omt_branching/solver/problem.py`
- Test: `tests/solver/test_problem.py`

**Interfaces:**
- Consumes: `Sense`, `GOMTState` from `interfaces`; a `SolveBackend`.
- Produces:
  - `@dataclass(frozen=True) class GOMTProblem`: `hard: Constraint`, `objective: Term`, `sense: Sense`.
    - `from_constraints(constraints: list, objective, sense) -> GOMTProblem` classmethod that conjoins via a backend-free marker — actually conjunction needs backend, so store `hard` already conjoined by caller, OR accept a list and let `initial_state` conjoin. Decision: store `hard` as a **list** `hard_list: tuple[Constraint, ...]`; `initial_state(backend)` conjoins.
  - `initial_state(self, backend) -> GOMTState`: `phi = backend.conjoin(*self.hard_list)`; `I0 = backend.solve(phi)`; if `None` raise `Infeasible`; `v0 = backend.value(I0, objective)`; `delta0 = backend.better(objective, v0, sense)`; returns `GOMTState(incumbent=I0, delta=delta0, tau=[delta0], objective=objective, sense=sense, hard=phi, stats={...0})`.
  - `class Infeasible(Exception)`.

- [ ] **Step 1: Write the failing test**

`tests/solver/test_problem.py`:
```python
from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")
from omt_branching.solver.interfaces import Sense
from omt_branching.solver.z3_backend import Z3Backend
from omt_branching.solver.problem import GOMTProblem, Infeasible


def test_initial_state_has_incumbent_and_better_delta():
    b = Z3Backend()
    x = z3.Int("x")
    prob = GOMTProblem(hard_list=(x >= 0, x <= 5), objective=x, sense=Sense.MIN)
    st = prob.initial_state(b)
    assert st.incumbent is not None
    # incumbent objective value is a feasible value in [0,5]
    v = b.value(st.incumbent, x)
    assert 0 <= v <= 5
    # delta excludes the incumbent's value (must be strictly better)
    assert b.solve(z3.And(st.hard, st.delta, x == v)) is None
    assert st.tau == [st.delta]


def test_infeasible_raises():
    b = Z3Backend()
    x = z3.Int("x")
    prob = GOMTProblem(hard_list=(x > 0, x < 0), objective=x, sense=Sense.MIN)
    with pytest.raises(Infeasible):
        prob.initial_state(b)
```

- [ ] **Step 2: Run to verify FAIL** — `conda run -n omt python -m pytest tests/solver/test_problem.py -v` → module not found.

- [ ] **Step 3: Write `problem.py`** per Produces. `GOMTProblem` is `frozen=True` with `hard_list: tuple`. Chinese docstrings. `__all__ = ["GOMTProblem", "Infeasible"]`.

- [ ] **Step 4: Run to verify PASS** — same command → 2 pass.

- [ ] **Step 5: Stage** — `git add omt_branching/solver/problem.py tests/solver/test_problem.py`

---

### Task 4: GOMT calculus engine (`calculus.py`)

**Files:**
- Create: `omt_branching/solver/calculus.py`
- Test: `tests/solver/test_calculus.py`

**Interfaces:**
- Consumes: `GOMTState`, `SplitDecision`, `Sense`, `SolveBackend`, `BranchingStrategy` from `interfaces`; `GOMTProblem` from `problem`.
- Produces:
  - `@dataclass(frozen=True) class GOMTConfig`: `max_steps: int = 10_000`, `f_sat_mode: str = "plain"` (`"plain"`|`"hybrid"`), `check_invariants: bool = False`.
  - `@dataclass class GOMTResult`: `model`, `value`, `optimal: bool`, `stats: dict`.
  - `class GOMTSolver`: `__init__(self, problem, backend, strategy, config=GOMTConfig())`. Method `run() -> GOMTResult`.
    - Build `state = problem.initial_state(backend)`. Loop while `not state.saturated and state.step < max_steps`:
      1. `decision = strategy.propose(state, backend)`.
      2. If `decision.kind == "split"`: **F-Split** — require `len(decision.subformulas) >= 1`; `state.tau = list(decision.subformulas) + state.tau[1:]`; `stats["splits"] += 1`. (Optional: if `check_invariants`, assert `backend.solve(backend.conjoin(state.hard, backend.negate(_iff(top_before, OR(subformulas))))) is None`.)
      3. Else (`"resolve"`): `psi = state.top`; `branch = backend.conjoin(state.hard, psi)`;
         - `plain`: `m = backend.solve(branch)`; if `m`: `v = backend.value(m, obj)` → **F-Sat**.
         - `hybrid`: `res = backend.optimize(branch, obj, sense)`; if `res`: `m, v = res` → **F-Sat**.
         - **F-Sat**: `state.incumbent = m`; `state.delta = backend.conjoin(state.delta, backend.better(obj, v, sense))`; `state.tau = [state.delta]`; `stats["sats"] += 1`.
         - If no model → **F-Close**: `state.delta = backend.conjoin(state.delta, backend.negate(psi))`; `state.tau = state.tau[1:]`; `stats["closes"] += 1`.
      4. `state.step += 1`.
    - On exit: `optimal = state.saturated`; `value = backend.value(state.incumbent, obj)`; return `GOMTResult(state.incumbent, value, optimal, stats)`.

- [ ] **Step 1: Write the failing test**

`tests/solver/test_calculus.py`:
```python
from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")
from omt_branching.solver.interfaces import Sense, SplitDecision
from omt_branching.solver.z3_backend import Z3Backend
from omt_branching.solver.problem import GOMTProblem
from omt_branching.solver.calculus import GOMTSolver, GOMTConfig
from tests.solver.conftest import random_lia_instance


class AlwaysResolve:
    """linear search 策略：从不 split，纯靠 F-Sat 收紧。"""
    def propose(self, state, backend):
        return SplitDecision.resolve()


def _native_opt(hard, obj, sense_min):
    o = z3.Optimize()
    for c in hard:
        o.add(c)
    h = o.minimize(obj) if sense_min else o.maximize(obj)
    assert o.check() == z3.sat
    ref = o.lower(h) if sense_min else o.upper(h)
    return ref.as_long()


def test_linear_search_reaches_native_optimum_min():
    b = Z3Backend()
    hard, xs, obj, _ = random_lia_instance(seed=1)
    prob = GOMTProblem(hard_list=tuple(hard), objective=obj, sense=Sense.MIN)
    res = GOMTSolver(prob, b, AlwaysResolve()).run()
    assert res.optimal is True
    assert res.value == _native_opt(hard, obj, sense_min=True)
    assert res.stats["sats"] >= 1


def test_split_path_with_objective_bisection():
    """手写一个 split 一次的策略，验证 F-Split 分支与 soundness。"""
    b = Z3Backend()
    x = z3.Int("x")
    prob = GOMTProblem(hard_list=(x >= 0, x <= 10), objective=x, sense=Sense.MAX)

    class SplitOnce:
        def __init__(self):
            self.done = False
        def propose(self, state, backend):
            if not self.done:
                self.done = True
                psi = state.top
                # ψ1 = ψ ∧ x>=6 (better half first), ψ2 = ψ ∧ x<6
                return SplitDecision.split([backend.conjoin(psi, backend.ge(x, 6)),
                                            backend.conjoin(psi, backend.le(x, 5))])
            return SplitDecision.resolve()

    res = GOMTSolver(prob, b, SplitOnce(),
                     GOMTConfig(check_invariants=True)).run()
    assert res.optimal is True
    assert res.value == 10
    assert res.stats["splits"] == 1
```

- [ ] **Step 2: Run to verify FAIL** — module not found.

- [ ] **Step 3: Write `calculus.py`** per Produces. Initialize `stats = {"splits": 0, "sats": 0, "closes": 0, "solve_calls": 0, "steps": 0}`. Helper `_iff(a, b)` only used when `check_invariants` (build via `z3`? NO — calculus must not import z3). Instead the invariant check uses backend only: equivalence of `top_before` and `OR(subformulas)` modulo hard. Implement OR through a new backend method? To avoid adding `disjoin`, fold the invariant check into the strategy/tests rather than calculus. **Decision:** drop the in-calculus invariant check; keep `check_invariants` flag reserved but no-op in v1 (documented). The construction (`ψ∧a`, `ψ∧¬a`) guarantees the premise; tests assert optimum equality which would fail if F-Split were unsound. `__all__ = ["GOMTSolver", "GOMTConfig", "GOMTResult"]`.

  > Note: this removes the only place that needed `_iff`. `check_invariants` stays as an accepted config field (default False) for forward-compat but performs no z3 work, preserving the "calculus does not import z3" constraint.

- [ ] **Step 4: Run to verify PASS** — 2 tests pass.

- [ ] **Step 5: Stage** — `git add omt_branching/solver/calculus.py tests/solver/test_calculus.py`

---

### Task 5: z3 snapshot extractor (`extractor.py`)

**Files:**
- Create: `omt_branching/solver/extractor.py`
- Test: `tests/solver/test_extractor.py`

**Interfaces:**
- Consumes: `GOMTState`, `Sense`; z3; `SolverSnapshot` and members from `omt_branching.input.solver_state`; `SearchMode` from `omt_branching.interfaces`.
- Produces:
  - `@dataclass class Handle`: `kind: str` (`"atom"`|`"numeric"`), `z3_obj: Term`, `var_id`, `current_value=None`, `lower=None`, `upper=None`. (carries enough to build a split.)
  - `@dataclass class Extraction`: `snapshot: SolverSnapshot`, `atom_handles: dict[Hashable, Handle]`, `numeric_handles: dict[Hashable, Handle]`.
  - `class Z3SnapshotExtractor`: `__init__(self, problem)` stores objective term/sense and pre-scans `problem` once. Method `extract(self, state, backend) -> Extraction`:
    - Walk `state.hard` AST collecting: numeric vars (z3 Int/Real consts), linear atoms (`<=,<,>=,>,=` whose args are linear over numeric vars) → `TheoryAtomInfo(var_coeffs, rhs, kind)` with a fresh `bool_var_id`; pure Boolean consts → `BooleanVarInfo`.
    - For each numeric var: `is_integer = z3.is_int_value`-style via sort; `lower_bound/upper_bound` from collected box atoms; `lp_value = backend.value(state.incumbent, var)` (current incumbent value — honest, not LP relaxation); `objective_coeff` from objective linear scan.
    - `objective`: `ObjectiveInfo(sense_is_min=(sense==MIN), incumbent=backend.value(I,obj), var_coeffs=...)`.
    - `search_state`: `SearchStateInfo(depth=state.step, decision_level=len(state.tau), conflict_count=state.stats.get("closes",0), search_mode=SearchMode.LINEAR)`.
    - `candidate_*`: all collected atom bool ids / numeric ids whose value is not yet pinned by `state.delta` (best-effort: include all).
    - Build `Handle`s mapping each atom's `bool_var_id`→z3 atom expr, each numeric `num_var_id`→z3 var (with current_value, lower, upper).
  - Helper `_linear_terms(expr) -> (dict[var, coeff], const)`: recursively decompose a z3 arithmetic expression into a coeff map + constant; raise/skip on non-linear.

- [ ] **Step 1: Write the failing test**

`tests/solver/test_extractor.py`:
```python
from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")
from omt_branching.solver.interfaces import Sense
from omt_branching.solver.z3_backend import Z3Backend
from omt_branching.solver.problem import GOMTProblem
from omt_branching.solver.extractor import Z3SnapshotExtractor


def test_extract_counts_and_coeffs():
    b = Z3Backend()
    x, y = z3.Int("x"), z3.Int("y")
    hard = (x >= 0, x <= 10, y >= 0, y <= 10, 2 * x + 3 * y <= 12)
    prob = GOMTProblem(hard_list=hard, objective=x + y, sense=Sense.MAX)
    st = prob.initial_state(b)
    ex = Z3SnapshotExtractor(prob).extract(st, b)
    snap = ex.snapshot
    # two numeric vars
    ids = {nv.num_var_id for nv in snap.numeric_vars}
    assert len(ids) == 2
    # the 2x+3y<=12 atom has coeffs {x:2, y:3}, rhs 12
    atoms = [a for a in snap.theory_atoms
             if set(a.var_coeffs.values()) == {2.0, 3.0}]
    assert len(atoms) == 1 and atoms[0].rhs == 12.0
    # objective coeffs both 1
    assert all(c == 1.0 for c in snap.objective.var_coeffs.values())
    assert snap.objective.sense_is_min is False
    # handles map back
    assert len(ex.numeric_handles) == 2


def test_extract_robust_on_incumbent_values():
    b = Z3Backend()
    x = z3.Int("x")
    prob = GOMTProblem(hard_list=(x >= 0, x <= 5), objective=x, sense=Sense.MIN)
    st = prob.initial_state(b)
    ex = Z3SnapshotExtractor(prob).extract(st, b)
    h = next(iter(ex.numeric_handles.values()))
    assert h.current_value is not None
```

- [ ] **Step 2: Run to verify FAIL** — module not found.

- [ ] **Step 3: Write `extractor.py`** per Produces. Use a stable id scheme: numeric var id = `str(var)` (z3 decl name); atom id = `f"atom{i}"` in AST traversal order; bool var id for an atom = `f"b_{atom_id}"`. `_linear_terms` handles `+`, `*` (const×var), unary minus, numerals; on unsupported node, record the var with coeff via fallback or skip the atom (log via `warnings.warn`). Fill `BooleanVarInfo.occurrence_count` best-effort (count atom appearances). `__all__ = ["Z3SnapshotExtractor", "Extraction", "Handle"]`.

- [ ] **Step 4: Run to verify PASS** — 2 tests pass.

- [ ] **Step 5: Stage** — `git add omt_branching/solver/extractor.py tests/solver/test_extractor.py`

---

### Task 6: Strategies (`strategy.py`)

**Files:**
- Create: `omt_branching/solver/strategy.py`
- Test: `tests/solver/test_strategy.py`

**Interfaces:**
- Consumes: `SplitDecision`, `GOMTState`, `Sense`, `SolveBackend`, `BranchingStrategy`; `Z3SnapshotExtractor`, `Handle` from `extractor`; `BranchingPolicyService` from `omt_branching.service`; `BranchingAdvice` from `omt_branching.output.advice`.
- Produces:
  - `@dataclass(frozen=True) class StrategyConfig`: `max_split_depth: int = 6`, `min_confidence: float = 0.0`.
  - `class BaselineStrategy(BranchingStrategy)`: objective binary-search. Tracks a per-`Top` split-depth counter via `state.stats`. `propose`: if depth on current branch `< max_split_depth` and the branch's objective range is splittable, return `split([ψ∧(t≺mid), ψ∧¬(t≺mid)])` (better half first via `backend.better`/`negate`); else `resolve()`. Bisection on the objective value bound derived from incumbent and a tracked opposite bound. Simpler robust version: split once per branch on `backend.better(obj, mid, sense)` where `mid` halves the gap between incumbent value and a stored extreme; fall back to `resolve()` when gap ≤ 1.
  - `class NeuralStrategy(BranchingStrategy)`: `__init__(self, problem, service: BranchingPolicyService, config=StrategyConfig())` builds a `Z3SnapshotExtractor`. `propose`:
    1. depth = `state.stats.get("branch_depth", 0)`; if depth ≥ `max_split_depth`: return `resolve()`.
    2. `ex = self.extractor.extract(state, backend)`; `advice = self.service.advise(ex.snapshot)`.
    3. if `advice.fallback` or `advice.confidence < min_confidence` or no candidate: return `resolve()`.
    4. translate via `_advice_to_split(advice, ex, state, backend)`:
       - if top candidate is an atom handle `h`: `a = h.z3_obj`; order by `advice.phase` (True→atom first): `psi=state.top`; `f_true=backend.conjoin(psi, a)`, `f_false=backend.conjoin(psi, backend.negate(a))`; subformulas = `[f_true, f_false]` if phase else `[f_false, f_true]`.
       - elif integer split on numeric handle `h`: `m = floor(h.current_value)`; `lo=backend.conjoin(psi, backend.le(h.z3_obj, m))`, `hi=backend.conjoin(psi, backend.ge(h.z3_obj, m+1))`; order by `advice` integer direction.
       - increment `state.stats["branch_depth"]`; return `SplitDecision.split([...])`.
  - Both reset `state.stats["branch_depth"]=0` is handled by calculus F-Sat (new branch). **Add:** calculus F-Sat/F-Close already replace `tau`; reset `branch_depth` there → **modify Task 4** note: in F-Sat and F-Close set `state.stats["branch_depth"] = 0`. (Captured here as a cross-task dependency; ensure calculus resets it.)

- [ ] **Step 1: Write the failing test**

`tests/solver/test_strategy.py`:
```python
from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")
from omt_branching.solver.interfaces import Sense
from omt_branching.solver.z3_backend import Z3Backend
from omt_branching.solver.problem import GOMTProblem
from omt_branching.solver.calculus import GOMTSolver, GOMTConfig
from omt_branching.solver.strategy import BaselineStrategy, NeuralStrategy
from omt_branching.service import BranchingPolicyService
from tests.solver.conftest import random_lia_instance


def _native_opt(hard, obj, sense_min):
    o = z3.Optimize()
    for c in hard:
        o.add(c)
    h = o.minimize(obj) if sense_min else o.maximize(obj)
    assert o.check() == z3.sat
    return (o.lower(h) if sense_min else o.upper(h)).as_long()


@pytest.mark.parametrize("seed", range(5))
def test_baseline_reaches_optimum(seed):
    b = Z3Backend()
    hard, xs, obj, _ = random_lia_instance(seed=seed)
    prob = GOMTProblem(hard_list=tuple(hard), objective=obj, sense=Sense.MIN)
    res = GOMTSolver(prob, b, BaselineStrategy()).run()
    assert res.optimal and res.value == _native_opt(hard, obj, True)


@pytest.mark.parametrize("seed", range(5))
def test_neural_untrained_reaches_optimum(seed):
    b = Z3Backend()
    hard, xs, obj, _ = random_lia_instance(seed=seed)
    prob = GOMTProblem(hard_list=tuple(hard), objective=obj, sense=Sense.MAX)
    service = BranchingPolicyService()                # untrained policy
    res = GOMTSolver(prob, b, NeuralStrategy(prob, service)).run()
    # soundness: optimum independent of policy quality
    assert res.optimal and res.value == _native_opt(hard, obj, False)
```

- [ ] **Step 2: Run to verify FAIL** — module not found.

- [ ] **Step 3: Apply the Task-4 cross-dependency**: edit `calculus.py` so F-Sat and F-Close both set `state.stats["branch_depth"] = 0`. Then write `strategy.py` per Produces. Guard all numeric conversions; on any extraction/advice exception, log `warnings.warn` and return `resolve()` (never crash the search). `__all__ = ["BaselineStrategy", "NeuralStrategy", "StrategyConfig"]`.

- [ ] **Step 4: Run to verify PASS** — `conda run -n omt python -m pytest tests/solver/test_strategy.py -v` → 10 pass.

- [ ] **Step 5: Stage** — `git add omt_branching/solver/strategy.py omt_branching/solver/calculus.py tests/solver/test_strategy.py`

---

### Task 7: Bridge facade (`bridge.py`) + Oracle test

**Files:**
- Create: `omt_branching/solver/bridge.py`
- Test: `tests/solver/test_oracle.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `@dataclass(frozen=True) class BridgeConfig`: `f_sat_mode: str = "plain"`, `max_steps: int = 10_000`, `strategy: str = "neural"` (`"neural"`|`"baseline"`).
  - `class NeuralGOMTSolver`: `__init__(self, service: Optional[BranchingPolicyService] = None, config=BridgeConfig())`. Method `solve(self, hard_list, objective, sense) -> GOMTResult`: builds `Z3Backend`, `GOMTProblem`, chosen strategy (`NeuralStrategy(prob, service or BranchingPolicyService())` or `BaselineStrategy()`), `GOMTSolver(... GOMTConfig(max_steps, f_sat_mode))`, returns `run()`.
  - Convenience `solve_native(hard_list, objective, sense) -> Value` using `Z3Backend.optimize` directly (reference oracle).

- [ ] **Step 1: Write the failing oracle test**

`tests/solver/test_oracle.py`:
```python
from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")
from omt_branching.solver.interfaces import Sense
from omt_branching.solver.bridge import NeuralGOMTSolver, BridgeConfig, solve_native
from tests.solver.conftest import random_lia_instance


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("sense", [Sense.MIN, Sense.MAX])
def test_neural_gomt_matches_native(seed, sense):
    hard, xs, obj, _ = random_lia_instance(seed=seed)
    native = solve_native(tuple(hard), obj, sense)
    res = NeuralGOMTSolver().solve(tuple(hard), obj, sense)
    assert res.optimal is True
    assert res.value == native


def test_hybrid_mode_also_matches():
    hard, xs, obj, _ = random_lia_instance(seed=3)
    native = solve_native(tuple(hard), obj, Sense.MAX)
    res = NeuralGOMTSolver(config=BridgeConfig(f_sat_mode="hybrid")).solve(
        tuple(hard), obj, Sense.MAX)
    assert res.value == native
```

- [ ] **Step 2: Run to verify FAIL** — module not found.

- [ ] **Step 3: Write `bridge.py`** per Produces. `__all__ = ["NeuralGOMTSolver", "BridgeConfig", "solve_native"]`.

- [ ] **Step 4: Run to verify PASS** — `conda run -n omt python -m pytest tests/solver/test_oracle.py -v` → 41 pass (20×2 + 1).

- [ ] **Step 5: Run the FULL solver suite** — `conda run -n omt python -m pytest tests/solver -v` → all green.

- [ ] **Step 6: Stage** — `git add omt_branching/solver/bridge.py tests/solver/test_oracle.py`

---

### Task 8: API docs, example, exports, requirements

**Files:**
- Create: `omt_branching/solver/API.md`
- Create: `examples/z3_demo.py`
- Modify: `omt_branching/solver/__init__.py` (exports)
- Modify: `requirements.txt` (add z3-solver)
- Modify: `omt_branching/__init__.py` (optional top-level re-export of `NeuralGOMTSolver`)
- Test: `tests/solver/test_demo_smoke.py`

**Interfaces:**
- Produces: public exports `NeuralGOMTSolver`, `BridgeConfig`, `GOMTProblem`, `Sense`, `Z3Backend`, `NeuralStrategy`, `BaselineStrategy`, `solve_native`.

- [ ] **Step 1: Write the failing smoke test**

`tests/solver/test_demo_smoke.py`:
```python
from __future__ import annotations
import subprocess, sys, pytest
pytest.importorskip("z3")


def test_demo_runs_and_splits():
    out = subprocess.run([sys.executable, "-m", "examples.z3_demo"],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert "optimum" in out.stdout.lower()
```

- [ ] **Step 2: Run to verify FAIL** — `examples/z3_demo.py` missing → returncode != 0.

- [ ] **Step 3: Write `examples/z3_demo.py`**

A runnable demo: build a small OMT(LIA) instance, solve with `NeuralGOMTSolver` (neural) and `BaselineStrategy`, and `solve_native`; print each optimum + stats (splits/sats/closes/steps), assert the three optima agree. Must print a line containing `optimum`.

- [ ] **Step 4: Write `omt_branching/solver/__init__.py` exports**, append `z3-solver>=4.15.4,<4.16` to `requirements.txt`, optionally re-export `NeuralGOMTSolver` from `omt_branching/__init__.py`.

- [ ] **Step 5: Write `omt_branching/solver/API.md`** (Chinese) covering:
  - 数据流图 + 三个桥接缝。
  - `SolveBackend` / `BranchingStrategy` Protocol 完整签名与语义。
  - GOMT calculus ⇄ 契约映射表（同 spec §2 表）。
  - "z3 能/不能提供"字段表（结构字段 vs theory-internal 缺省）。
  - 最小集成示例（10 行：构造实例 → `NeuralGOMTSolver().solve(...)`）。
  - 扩展点：多目标 / lexicographic（`Sense`/`better` 可插拔位置）。

- [ ] **Step 6: Run demo + smoke test**

Run: `conda run -n omt python -m examples.z3_demo` (manual visual check: optima agree, splits>0)
Run: `conda run -n omt python -m pytest tests/solver/test_demo_smoke.py -v`
Expected: PASS.

- [ ] **Step 7: Final full suite** — `conda run -n omt python -m pytest tests/solver -v` → all green.

- [ ] **Step 8: Stage** — `git add omt_branching/solver/ examples/z3_demo.py requirements.txt omt_branching/__init__.py tests/solver/`

---

## Self-Review

**Spec coverage:**
- §1 scope (single-obj LIA/LRA, domain split, honest extraction) → Tasks 2–7, asserted in oracle test. ✓
- §2 GOMT calculus (F-Split/Sat/Close, Better, Optimize) → Task 4 + Task 2. ✓
- §3.1 two Protocols → Task 1. ✓
- §3.2 three seams (extractor / advise / advice-to-split) → Tasks 5,6. ✓
- §3.3 plain/hybrid F-Sat → Task 4 config + Task 7 hybrid test. ✓
- §4 module list (7 modules) → Tasks 1–7; API.md/example → Task 8. ✓
- §6 test strategy (oracle consistency, F-Close, fallback, extractor robustness, e2e) → oracle test (Task 7), strategy/extractor tests, demo smoke. ✓
- §7 conventions (Chinese docs, frozen dataclass, z3 isolation) → Global Constraints + each task. ✓

**Placeholder scan:** No "TBD/handle edge cases" left; the one removed item (in-calculus invariant check) is explicitly resolved to a documented no-op to preserve "calculus must not import z3."

**Type consistency:** `SplitDecision.split/resolve`, `GOMTState` fields, `Z3Backend` 10 methods, `GOMTResult`/`GOMTConfig`, `Extraction`/`Handle`, strategy ctors — names used in later tasks match their definitions. `state.stats["branch_depth"]` reset is explicitly threaded as a Task-4↔Task-6 dependency (Task 6 Step 3 edits calculus).

**Known cross-task edit:** Task 6 modifies `calculus.py` (Task 4) to reset `branch_depth` on F-Sat/F-Close. Flagged in both tasks.
