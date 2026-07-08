# UserPropagator 学习分支 Phase 2b（RL）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 RL（含 defer-to-VSIDS 动作）在更难实例上训练 decide 策略，让 learned-decide 在
rlimit/conflicts 上优于 VSIDS-decide（等 `== native`）。

**Architecture:** 每个 decide 对 `[defer, 未定原子分数]` softmax 采样——采到 defer 则退回 VSIDS
(return None)，否则覆盖采样原子。奖励 = −log1p(rlimit)，per-instance EMA baseline，REINFORCE
重算记录的 refocus 图。更难实例提供 headroom。imitation(Phase 2a) 作冷启动。

**Tech Stack:** Python 3.10 / z3-solver 4.15.4 / PyTorch；pytest。

## Global Constraints

- 运行/测试：`conda run -n omt python -m pytest tests/solver`（从 `Branching/`）。
- 不改 z3；decide 经 propagator `next_split`（硬覆盖）或 return None（退回 VSIDS）。
- docstring/注释中文；标识符/类型英文。
- 复用：`solve_omt_with_decider(hard, obj, sense, decider_factory, max_iters) -> dict{value,rlimit,conflicts,decisions,iters}`、`build_bool_snapshot`、`GraphBuilder(DEFAULT_FEATURE_SPEC).build(snap)`、`BranchingPolicy.infer(g)/__call__(g) -> PolicyOutput(bool_branch_scores: Tensor[num_bool])`、`graph.id_maps[NodeType.BOOL_VAR]: {atom_key->local}`、`generate_bool_lia_dataset`、rl.py 的 per-instance baseline 模式。
- 诚实：若更难实例仍对 VSIDS 太易（无 headroom），RL 会收敛到"总 defer"（平局非胜）——如实报告。

---

### Task 1: 更难布尔结构实例（headroom）

**Files:**
- Modify: `omt_branching/solver/instance_gen.py`
- Test: `tests/solver/test_instance_gen_lia.py`（追加）

**Interfaces:**
- Consumes: `generate_bool_lia_instance`（Phase 1）。
- Produces: `generate_hard_bool_lia_dataset(count, seed=0, *, min_vars=6, max_vars=8, **kw) -> list[OMTInstance]`（更多析取/更大子句/更紧池 -> VSIDS 冲突更多）。

- [ ] **Step 1: 写失败测试**

```python
# tests/solver/test_instance_gen_lia.py 追加
def test_hard_bool_lia_more_disjunctions_and_feasible():
    from omt_branching.solver.instance_gen import generate_hard_bool_lia_dataset, _validate
    ds = generate_hard_bool_lia_dataset(4, seed=1, min_vars=6, max_vars=6)
    assert len(ds) == 4
    for inst in ds:
        assert _validate(inst)                        # witness 驱动 -> SAT
        n_disj = sum(1 for a in inst.hard if a.decl().kind().__class__ and "or" in str(a.decl()).lower())
        # 更难：析取子句数明显多于变量数
        import z3
        n_or = sum(1 for a in inst.hard if z3.is_or(a))
        assert n_or >= 16
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_instance_gen_lia.py::test_hard_bool_lia_more_disjunctions_and_feasible -q`
Expected: FAIL（`ImportError: generate_hard_bool_lia_dataset`）

- [ ] **Step 3: 写实现**

在 `instance_gen.py` 的 `generate_bool_lia_dataset` 之后新增：

```python
def generate_hard_bool_lia_dataset(count: int, seed: int = 0, *, id_prefix: str = "hblia",
                                   min_vars: int = 6, max_vars: int = 8, **kwargs) -> list[OMTInstance]:
    """更难的布尔结构整数 OMT：更多析取(n_disj=24)、更大子句(k=4)、更紧原子池(pool_mult=1)，
    使 z3 VSIDS 探索更多冲突 -> 给学习分支留 headroom。"""
    hard_defaults = dict(n_disj=24, k=4, ub=10, chi=5, pool_mult=1)
    hard_defaults.update(kwargs)
    rng = random.Random(seed)
    out: list[OMTInstance] = []
    for i in range(count):
        n_vars = rng.randint(min_vars, max_vars)
        out.append(generate_bool_lia_instance(f"{id_prefix}{i}", rng, n_vars=n_vars, **hard_defaults))
    return out
```

在 `__all__` 加入 `"generate_hard_bool_lia_dataset"`；并在 `omt_branching/solver/__init__.py` 的
instance_gen import 与 `__all__` 加入它（与 `generate_bool_lia_dataset` 并列）。

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_instance_gen_lia.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/instance_gen.py omt_branching/solver/__init__.py tests/solver/test_instance_gen_lia.py
git commit -m "feat: generate_hard_bool_lia_dataset（更难布尔实例，给 RL headroom）"
```

---

### Task 2: SamplingPolicyDecider（采样 + defer，记录轨迹）

**Files:**
- Create: `omt_branching/solver/rl_decide.py`
- Test: `tests/solver/test_rl_decide.py`

**Interfaces:**
- Consumes: `build_bool_snapshot`, `GraphBuilder`, `BranchingPolicy.infer`, `NodeType`。
- Produces: `SamplingPolicyDecider(policy, defer_logit, assertions, refocus_every=50, sample=True)`：可调用 `(undecided_keys, assignment) -> Optional[tuple[str,bool]]`；属性 `.steps: list[tuple[HeteroGraph, list[int], int]]`（(refocus 图, 未定原子局部索引, 采样索引; 索引 0=defer)）。

- [ ] **Step 1: 写失败测试**

```python
# tests/solver/test_rl_decide.py
from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")
torch = pytest.importorskip("torch")

from omt_branching.model.policy import BranchingPolicy
from omt_branching.solver.rl_decide import SamplingPolicyDecider
from omt_branching.solver.propagator_snapshot import atom_key


def test_sampling_decider_records_steps_and_valid_choice():
    x = z3.Int("x")
    a, b = x >= 5, x <= 2
    asserts = [x >= 0, x <= 10, z3.Or(a, b)]
    policy = BranchingPolicy()
    defer = torch.zeros(())
    dec = SamplingPolicyDecider(policy, defer, asserts, refocus_every=100, sample=True)
    und = [atom_key(a), atom_key(b)]
    torch.manual_seed(0)
    outs = [dec(und, {}) for _ in range(5)]
    # 每次返回 None(defer) 或 合法未定原子+bool
    assert all(o is None or (o[0] in und and isinstance(o[1], bool)) for o in outs)
    assert len(dec.steps) == 5            # 记录了 5 步
    g, ls, idx = dec.steps[0]
    assert 0 <= idx <= len(ls)            # idx=0=defer, 1..len=原子
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_rl_decide.py -q`
Expected: FAIL（`ModuleNotFoundError: rl_decide`）

- [ ] **Step 3: 写实现**

```python
# omt_branching/solver/rl_decide.py
"""Decide 层 RL：采样式 decider（含 defer-to-VSIDS 动作）+ REINFORCE 训练器。

每个 decide 对 ``[defer_logit, 未定原子 bool 分数]`` softmax 采样：采到 defer -> return None
(退回 VSIDS)，否则覆盖采样原子。记录 (refocus 图, 未定局部索引, 采样索引) 供 REINFORCE 重算
log-prob。奖励 = −log1p(rlimit)，per-instance EMA baseline。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from omt_branching.input.graph_builder import DEFAULT_FEATURE_SPEC, GraphBuilder
from omt_branching.interfaces import NodeType
from omt_branching.model.policy import BranchingPolicy
from omt_branching.solver.decide_omt import solve_omt_with_decider
from omt_branching.solver.interfaces import Sense
from omt_branching.solver.propagator_snapshot import build_bool_snapshot


class SamplingPolicyDecider:
    def __init__(self, policy: BranchingPolicy, defer_logit, assertions,
                 refocus_every: int = 50, sample: bool = True):
        self.policy = policy
        self.defer_logit = defer_logit          # torch 标量（trainer 持有的可学参数）
        self.assertions = list(assertions)
        self.refocus_every = max(1, refocus_every)
        self.sample = sample
        self._graph = None
        self._scores = None                      # detached bool_branch_scores
        self._bmap: dict = {}
        self._since = self.refocus_every
        self.steps: list = []

    def _refocus(self, assignment):
        snap, _ = build_bool_snapshot(self.assertions, assignment=assignment)
        g = GraphBuilder(DEFAULT_FEATURE_SPEC).build(snap)
        out = self.policy.infer(g)
        self._graph = g
        self._scores = out.bool_branch_scores.detach()
        self._bmap = g.id_maps.get(NodeType.BOOL_VAR, {})

    def __call__(self, undecided_keys, assignment) -> Optional[tuple]:
        if self._since >= self.refocus_every:
            self._refocus(assignment)
            self._since = 0
        self._since += 1
        if self._graph is None or self._scores is None or self._scores.numel() == 0:
            return None
        pairs = [(k, self._bmap.get(k)) for k in undecided_keys]
        pairs = [(k, l) for k, l in pairs if l is not None and l < self._scores.numel()]
        if not pairs:
            return None
        keys = [k for k, _ in pairs]
        locs = [l for _, l in pairs]
        logits = torch.cat([self.defer_logit.detach().reshape(1), self._scores[locs]])
        probs = torch.softmax(logits, dim=0)
        idx = int(torch.multinomial(probs, 1).item()) if self.sample else int(torch.argmax(probs).item())
        self.steps.append((self._graph, locs, idx))
        if idx == 0:
            return None                          # defer -> VSIDS
        return keys[idx - 1], True               # 覆盖采样原子（相位取真）


@dataclass
class DecideRLConfig:
    lr: float = 1e-3
    refocus_every: int = 50
    max_iters: int = 100000
    baseline_momentum: float = 0.9
    grad_clip: float = 5.0
    device: str = "cpu"


class DecideRLTrainer:
    def __init__(self, policy: BranchingPolicy, config: DecideRLConfig = DecideRLConfig()):
        self.policy = policy.to(config.device)
        self.config = config
        self.defer_logit = torch.nn.Parameter(torch.zeros((), device=config.device))
        self.opt = torch.optim.Adam(list(policy.parameters()) + [self.defer_logit], lr=config.lr)
        self._baselines: dict = {}
        self._baseline = 0.0

    def _baseline_for(self, key):
        return self._baselines.get(key, self._baseline)

    def _update_baseline_for(self, key, value):
        m = self.config.baseline_momentum
        if key in self._baselines:
            self._baselines[key] = m * self._baselines[key] + (1 - m) * value
        else:
            self._baselines[key] = value

    def collect(self, hard, objective, sense: Sense):
        holder: dict = {}

        def factory(assertions):
            d = SamplingPolicyDecider(self.policy, self.defer_logit, assertions,
                                      self.config.refocus_every, sample=True)
            holder["d"] = d
            return d

        res = solve_omt_with_decider(hard, objective, sense,
                                     decider_factory=factory, max_iters=self.config.max_iters)
        steps = holder["d"].steps if "d" in holder else []
        reward = -math.log1p(res["rlimit"])
        return steps, reward, res

    def update(self, steps, reward, key) -> dict:
        if not steps:
            self._update_baseline_for(key, reward)
            return {"loss": 0.0, "reward": reward, "steps": 0}
        adv = reward - self._baseline_for(key)
        cache: dict = {}
        loss = torch.zeros((), device=self.config.device)
        n = 0
        for g, locs, idx in steps:
            gid = id(g)
            if gid not in cache:
                cache[gid] = self.policy(g).bool_branch_scores
            scores = cache[gid]
            logits = torch.cat([self.defer_logit.reshape(1), scores[locs]])
            logp = torch.log_softmax(logits, dim=0)[idx]
            loss = loss - logp * adv
            n += 1
        loss = loss / n
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()) + [self.defer_logit],
                                       self.config.grad_clip)
        self.opt.step()
        self._update_baseline_for(key, reward)
        return {"loss": float(loss), "reward": reward, "steps": n}

    def train(self, instances, iterations: int = 1, log: bool = False):
        instances = list(instances)
        history = []
        for it in range(iterations):
            for j, (hard, obj, sense) in enumerate(instances):
                steps, reward, res = self.collect(hard, obj, sense)
                stats = self.update(steps, reward, key=j)
                stats.update({"iter": it, "instance": j, "rlimit": res["rlimit"],
                              "conflicts": res["conflicts"]})
                history.append(stats)
                if log:
                    print(f"[it {it} inst {j}] loss={stats['loss']:.4f} reward={reward:.3f} "
                          f"rlimit={res['rlimit']} conflicts={res['conflicts']} steps={stats['steps']}")
        return history


__all__ = ["SamplingPolicyDecider", "DecideRLConfig", "DecideRLTrainer"]
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_rl_decide.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/rl_decide.py tests/solver/test_rl_decide.py
git commit -m "feat: SamplingPolicyDecider + DecideRLTrainer（采样+defer，REINFORCE）"
```

---

### Task 3: DecideRLTrainer collect+update 冒烟

**Files:**
- Test: `tests/solver/test_rl_decide.py`（追加）

**Interfaces:**
- Consumes: `DecideRLTrainer`（Task 2）、`generate_bool_lia_dataset`。
- Produces: 无新代码——端到端护栏（一次 collect+update 跑通、loss 有限、baseline 更新、reward 计算）。

- [ ] **Step 1: 写测试**

```python
# tests/solver/test_rl_decide.py 追加
def test_decide_rl_collect_update_runs():
    from omt_branching.solver import generate_bool_lia_dataset
    from omt_branching.solver.rl_decide import DecideRLTrainer, DecideRLConfig
    import math

    inst = generate_bool_lia_dataset(1, seed=3, min_vars=5, max_vars=5)[0]
    hard, obj, sense = inst.as_tuple()
    tr = DecideRLTrainer(BranchingPolicy(), DecideRLConfig(refocus_every=30))
    steps, reward, res = tr.collect(hard, obj, sense)
    assert res["value"] is not None and res["rlimit"] > 0
    assert math.isfinite(reward)
    stats = tr.update(steps, reward, key=0)
    assert math.isfinite(stats["loss"])
    assert 0 in tr._baselines                    # baseline 记录
```

- [ ] **Step 2: 运行**

Run: `conda run -n omt python -m pytest tests/solver/test_rl_decide.py -q`
Expected: PASS（2 passed）。若 collect 很慢，减小 refocus/实例。

- [ ] **Step 3: （无实现——护栏）**

- [ ] **Step 4: 提交**

```bash
git add tests/solver/test_rl_decide.py
git commit -m "test: DecideRLTrainer collect+update 端到端护栏"
```

---

### Task 4: 实验加 --rl-iters + 测量 RL vs VSIDS

**Files:**
- Modify: `examples/decide_branch.py`
- Test: `tests/solver/test_demo_smoke.py`（现有 decide_branch 冒烟无需改）

**Interfaces:**
- Consumes: `DecideRLTrainer`、`generate_hard_bool_lia_dataset`、Phase 2a 的 imitation。
- Produces: `decide_branch.py` 支持 `--rl-iters R`（imitation 冷启动后 RL 微调）与 `--hard`（用更难实例）。

- [ ] **Step 1: 改实验脚本**

`decide_branch.py` argparse 增：
```python
    ap.add_argument("--rl-iters", type=int, default=0, help="RL 微调轮数(0=不做 RL)")
    ap.add_argument("--hard", action="store_true", help="用更难实例(headroom)")
```
数据集选择改为：
```python
    from omt_branching.solver import generate_hard_bool_lia_dataset
    gen = generate_hard_bool_lia_dataset if args.hard else generate_bool_lia_dataset
    insts = gen(args.test, seed=99, min_vars=args.min_vars, max_vars=args.max_vars)
```
（训练集同样用 `gen(...)`。）在 imitation 训练块之后、`svc=` 之前插入 RL：
```python
    if args.rl_iters > 0:
        from omt_branching.solver.rl_decide import DecideRLTrainer, DecideRLConfig
        rl_train = gen(max(args.train, 40), seed=1, min_vars=args.min_vars, max_vars=args.max_vars)
        rlt = DecideRLTrainer(policy, DecideRLConfig(refocus_every=args.refocus))
        h = rlt.train([i.as_tuple() for i in rl_train], iterations=args.rl_iters, log=False)
        if h:
            print(f"RL 微调: {len(h)} 步, 末条 reward={h[-1]['reward']:.3f} "
                  f"conflicts={h[-1]['conflicts']}, defer_logit={float(rlt.defer_logit):.3f}")
```

- [ ] **Step 2: 跑实验测量（关键经验验证）**

Run: `conda run -n omt python -m examples.decide_branch --hard --test 20 --train 40 --epochs 20 --rl-iters 2 --min-vars 6 --max-vars 7`
Expected: 打印 imitation loss、RL reward/defer_logit、三臂对比。**关键看**：RL 后 learned-decide 的
conflicts/rlimit 是否 **< VSIDS**（成功判据）；`match=1` 必须保持。
（诚实：若仍 ≥ VSIDS，记录数字 + defer_logit（若 defer_logit 很大 = RL 学会"总退回 VSIDS"=无
headroom 的证据）——如实报告，不粉饰。）

- [ ] **Step 3: 全量**

Run: `conda run -n omt python -m pytest tests/solver -q`
Expected: 全绿。

- [ ] **Step 4: 提交**

```bash
git add examples/decide_branch.py
git commit -m "feat: decide_branch 加 --rl-iters/--hard + 测量 RL vs VSIDS"
```

---

## Self-Review

**Spec 覆盖（Phase 2 §2.3 RL）：** RL 采样 decide + defer → Task 2 ✅；reward=−rlimit + per-instance
baseline + REINFORCE → Task 2/3 ✅；更难实例(headroom) → Task 1 ✅；实验测量 → Task 4 ✅。

**Placeholder scan：** 无 TBD/TODO；每步含完整代码。

**Type consistency：** `SamplingPolicyDecider(policy, defer_logit, assertions, refocus_every, sample)`
+ `.steps: list[(graph, locs, idx)]` Task 2 定义、Task 3 collect 消费一致；`DecideRLTrainer(policy, config)`
的 `collect/update/train` 签名 Task 2/3/4 一致；`generate_hard_bool_lia_dataset(count, seed, min_vars, max_vars, **kw)` Task 1 定义、Task 4 调用一致。

**风险提示：** 若 Task 4 RL 未胜 VSIDS——检查 defer_logit：若很大=RL 学会"总退回 VSIDS"(平局)=实例
仍无 headroom → 需更难实例/别的分支族；若适中但仍劣=RL 未收敛/reward 塑形。诚实报告，不强求胜。
