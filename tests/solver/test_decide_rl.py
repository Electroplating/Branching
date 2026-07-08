from __future__ import annotations
import math

import pytest
z3 = pytest.importorskip("z3")
torch = pytest.importorskip("torch")

from omt_branching.model.policy import BranchingPolicy
from omt_branching.solver import generate_bool_lia_dataset, solve_native
from omt_branching.solver.decide_rl import (
    DecideRLConfig, DecideRLTrainer, SamplingDecider,
)
from omt_branching.solver.propagator_snapshot import atom_key


def test_sampling_decider_records_steps():
    x = z3.Int("x")
    a, b = x >= 5, x <= 2
    asserts = [x >= 0, x <= 10, z3.Or(a, b)]
    dec = SamplingDecider(BranchingPolicy(), asserts, DecideRLConfig(refocus_every=100),
                          sample=True)
    und = [atom_key(a), atom_key(b)]
    choice = dec(und, {})
    assert choice is not None
    assert choice[0] in und and isinstance(choice[1], bool)
    assert len(dec.steps) == 1        # 首次调用触发一次 refocus -> 记录一步


def test_collect_episode_matches_native_and_updates():
    inst = generate_bool_lia_dataset(1, seed=5, min_vars=4, max_vars=4)[0]
    hard, obj, sense = inst.as_tuple()
    trainer = DecideRLTrainer(BranchingPolicy(), DecideRLConfig(refocus_every=20))
    ep = trainer.collect_episode(hard, obj, sense)
    assert ep.value == solve_native(hard, obj, sense)   # 正确性：== native
    assert ep.rlimit > 0
    assert ep.decisions is not None                     # propagator 生效
    stats = trainer.update(ep, key=0)
    assert math.isfinite(stats["loss"])


def test_train_and_evaluate():
    insts = generate_bool_lia_dataset(2, seed=8, min_vars=4, max_vars=4)
    instances = [i.as_tuple() for i in insts]
    trainer = DecideRLTrainer(BranchingPolicy(), DecideRLConfig(refocus_every=20))
    history = trainer.train(instances, iterations=1, log=False)
    assert len(history) == 2
    hard, obj, sense = instances[0]
    res = trainer.evaluate(hard, obj, sense)
    assert res["value"] == solve_native(hard, obj, sense)
    assert res["rlimit"] > 0
