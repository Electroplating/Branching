from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")
torch = pytest.importorskip("torch")

from omt_branching.model.policy import BranchingPolicy
from omt_branching.model.trainer import ImitationTrainer, TrainConfig
from omt_branching.solver.instance_gen import generate_bool_lia_dataset
from omt_branching.solver.lookahead import build_lookahead_examples


def test_imitation_coldstart_reduces_loss():
    """look-ahead 标签冷启动：ImitationTrainer 应能显著降低 bool/phase 损失。"""
    torch.manual_seed(0)
    insts = generate_bool_lia_dataset(6, seed=11, min_vars=4, max_vars=5)
    examples = build_lookahead_examples([list(i.hard) for i in insts])
    assert examples, "应能构造 look-ahead 样本"

    policy = BranchingPolicy()
    trainer = ImitationTrainer(policy, TrainConfig(lr=5e-3))
    history = trainer.fit(examples, epochs=15)
    assert history[-1]["loss"] < history[0]["loss"]     # 收敛：末轮损失更低
    assert "branch" in history[-1]                        # bool head 参与了训练
