"""阶段一：离线 imitation learning (plan 8.1)。

用专家标签（strong branching / oracle subtree / heuristic 蒸馏）训练 ranking 策略。
主损失是候选变量上的 **soft 交叉熵 / KL**（ListNet 风格），辅以 phase BCE、整数
B&B ranking、以及多任务辅助损失:

``L = L_branch + λ1 L_phase + λ2 L_int + λ3 L_aux``  (plan 5.3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from omt_branching.model.device import gnn_device
from omt_branching.graph.hetero_graph import HeteroGraph
from omt_branching.model.policy import BranchingPolicy, PolicyOutput

from tqdm import tqdm

@dataclass
class RankingExample:
    """单个训练样本：一张图 + 专家标签。

    标签均以 **图内局部索引** 给出（由 ``GraphBuilder`` 的 id_maps 决定）。
    各字段可选，缺失则对应损失项不计入。
    """

    graph: HeteroGraph
    # 布尔 branching 标签：local_idx -> 专家分数 / 偏好权重 (越大越该选)
    bool_target_scores: dict[int, float] = field(default_factory=dict)
    # phase 标签：local_idx -> 取真(True)/取假(False)
    phase_targets: dict[int, bool] = field(default_factory=dict)
    # defer 教师分（与原子分同一 ListNet；对应 SamplingPolicyDecider 的 defer_logit）
    defer_target_score: float | None = None
    # 整数 B&B 标签：local_idx -> 偏好权重；方向 local_idx -> 向上(True)
    int_target_scores: dict[int, float] = field(default_factory=dict)
    int_dir_targets: dict[int, bool] = field(default_factory=dict)
    # 辅助任务标签：local_idx -> 值
    conflict_targets: dict[int, bool] = field(default_factory=dict)
    core_targets: dict[int, bool] = field(default_factory=dict)
    obj_improve_targets: dict[int, float] = field(default_factory=dict)
    subtree_targets: dict[int, float] = field(default_factory=dict)


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    lambda_phase: float = 0.5
    lambda_int: float = 1.0
    lambda_aux: float = 0.3
    grad_clip: float = 5.0
    device: str = field(default_factory=gnn_device)
    accum_steps: int = 1  # 梯度累积步数，>1 时每 accum_steps 次 forward 才 backward 一次
    # epochs=-1 时早停
    patience: int = 5
    tol: float = 0.01  # branch（或 total loss）相对提升阈值
    max_epochs: int = 10_000


class ImitationTrainer:
    """封装优化器与多任务损失；可选学习 ``defer_logit``（与 RL 同构）。"""

    def __init__(self, policy: BranchingPolicy, config: TrainConfig = TrainConfig()):
        self.policy = policy.to(config.device)
        self.config = config
        self.defer_logit = torch.nn.Parameter(
            torch.zeros((), dtype=torch.float32, device=config.device)
        )
        self.opt = torch.optim.Adam(
            list(policy.parameters()) + [self.defer_logit],
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    # ------------------------------------------------------------------ #
    def train_step(self, example: RankingExample, backward: bool = True) -> dict[str, float] | tuple[dict[str, float], torch.Tensor]:
        self.policy.train()
        g = example.graph.to(self.config.device)
        out = self.policy(g)
        loss, parts = self._compute_loss(out, example)

        if backward:
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + [self.defer_logit],
                self.config.grad_clip,
            )
            self.opt.step()
            return parts
        return parts, loss

    def fit(
        self,
        examples: Iterable[RankingExample],
        epochs: int = 1,
        log_every: int = 0,
        *,
        patience: int | None = None,
        tol: float | None = None,
        max_epochs: int | None = None,
    ) -> list[dict[str, float]]:
        """训练 ``epochs`` 轮；``epochs=-1`` 时训到 branch/loss 相对提升不足为止。"""
        examples = list(examples)
        if not examples:
            return []
        history: list[dict[str, float]] = []
        pat = self.config.patience if patience is None else int(patience)
        tol_v = self.config.tol if tol is None else float(tol)
        cap = self.config.max_epochs if max_epochs is None else int(max_epochs)
        if epochs == -1:
            n_epochs = max(1, cap)
            early = True
        else:
            n_epochs = max(1, int(epochs))
            early = False

        best = None
        stall = 0
        pbar_total = None if early else n_epochs * len(examples)
        with tqdm(total=pbar_total, desc="imit_train") as pbar:
            for ep in range(n_epochs):
                agg: dict[str, float] = {}
                self.opt.zero_grad()
                for i, ex in enumerate(examples):
                    parts, loss_tensor = self.train_step(ex, backward=False)
                    (loss_tensor / self.config.accum_steps).backward()
                    for k, v in parts.items():
                        agg[k] = agg.get(k, 0.0) + v
                    if (i + 1) % self.config.accum_steps == 0 or (i + 1) == len(examples):
                        torch.nn.utils.clip_grad_norm_(
                            list(self.policy.parameters()) + [self.defer_logit],
                            self.config.grad_clip,
                        )
                        self.opt.step()
                        self.opt.zero_grad()
                    if log_every and (i + 1) % log_every == 0:
                        print(f"[epoch {ep} step {i + 1}] loss={parts['loss']:.4f}")
                    pbar.update(1)
                n = max(1, len(examples))
                row = {k: v / n for k, v in agg.items()}
                history.append(row)
                metric = float(row.get("branch", row.get("loss", 0.0)))
                if early:
                    if best is None:
                        best = metric
                        stall = 0
                    else:
                        scale = abs(best) + 1e-8
                        if (best - metric) / scale > tol_v:
                            best = metric
                            stall = 0
                        else:
                            stall += 1
                    pbar.set_postfix(
                        ep=ep + 1,
                        branch=f"{metric:.4f}",
                        best=f"{best:.4f}",
                        stall=f"{stall}/{pat}",
                    )
                    if stall >= pat:
                        history.append(
                            {
                                "event": "early_stop",
                                "epoch": ep + 1,
                                "best_branch": best,
                                "patience": pat,
                                "tol": tol_v,
                            }
                        )
                        break
                else:
                    pbar.set_postfix(ep=ep + 1, branch=f"{metric:.4f}")
        return history

    # ------------------------------------------------------------------ #
    def _compute_loss(self, out: PolicyOutput, ex: RankingExample):
        dev = self.config.device
        parts: dict[str, float] = {}
        total = torch.zeros((), device=dev)

        # --- 布尔 branching ranking (soft CE / KL)，可选拼接 defer ---
        l_branch = _ranking_loss(
            out.bool_branch_scores,
            ex.bool_target_scores,
            out.candidate_bool_local,
            dev,
            defer_logit=self.defer_logit,
            defer_target=ex.defer_target_score,
        )
        if l_branch is not None:
            total = total + l_branch
            parts["branch"] = float(l_branch.detach())

        # --- phase BCE ---
        l_phase = _bce_loss(out.phase_logits, ex.phase_targets, dev)
        if l_phase is not None:
            total = total + self.config.lambda_phase * l_phase
            parts["phase"] = float(l_phase)

        # --- 整数 B&B ranking + 方向 ---
        l_int = _ranking_loss(out.int_branch_scores, ex.int_target_scores,
                              out.candidate_numeric_local, dev)
        if l_int is not None:
            total = total + self.config.lambda_int * l_int
            parts["int"] = float(l_int)
        l_dir = _bce_loss(out.int_dir_logits, ex.int_dir_targets, dev)
        if l_dir is not None:
            total = total + self.config.lambda_int * l_dir
            parts["int_dir"] = float(l_dir)

        # --- 辅助多任务 ---
        aux = out.aux
        l_aux = torch.zeros((), device=dev)
        used = False
        if aux:
            for key, targets, kind in [
                ("conflict_logit", ex.conflict_targets, "bce"),
                ("in_core_logit", ex.core_targets, "bce"),
                ("obj_improve", ex.obj_improve_targets, "mse"),
                ("subtree_size", ex.subtree_targets, "mse"),
            ]:
                if not targets or key not in aux:
                    continue
                if kind == "bce":
                    li = _bce_loss(aux[key], targets, dev)
                else:
                    li = _mse_loss(aux[key], targets, dev)
                if li is not None:
                    l_aux = l_aux + li
                    used = True
        if used:
            total = total + self.config.lambda_aux * l_aux
            parts["aux"] = float(l_aux)

        parts["loss"] = float(total)
        parts["defer_logit"] = float(self.defer_logit.detach().cpu())
        return total, parts


# --------------------------------------------------------------------------- #
# 损失工具
# --------------------------------------------------------------------------- #
def _ranking_loss(
    scores: torch.Tensor,
    target_scores: dict[int, float],
    candidate_local: list[int],
    dev,
    *,
    defer_logit: torch.Tensor | None = None,
    defer_target: float | None = None,
) -> Optional[torch.Tensor]:
    """候选集合上的 soft 交叉熵 (ListNet)：KL(target || softmax(scores))。

    若提供 ``defer_target``，在 logits/目标前拼接 defer（与 RL SamplingPolicyDecider 对齐）。
    """
    if scores.numel() == 0 or not target_scores:
        return None
    cand = candidate_local or sorted(target_scores.keys())
    cand = [c for c in cand if 0 <= c < scores.numel()]
    if not cand:
        return None
    idx = torch.tensor(cand, dtype=torch.long, device=dev)
    logits = scores[idx]
    tgt = torch.tensor([target_scores.get(c, 0.0) for c in cand], device=dev)
    if defer_target is not None and defer_logit is not None:
        logits = torch.cat([defer_logit.reshape(1).to(dev), logits])
        tgt = torch.cat(
            [torch.tensor([float(defer_target)], device=dev), tgt]
        )
    if float(tgt.abs().sum()) == 0.0:
        return None
    tgt = F.softmax(tgt, dim=0)
    logp = F.log_softmax(logits, dim=0)
    return -(tgt * logp).sum()


def _bce_loss(logits: torch.Tensor, targets: dict[int, bool], dev) -> Optional[torch.Tensor]:
    if logits.numel() == 0 or not targets:
        return None
    keys = [k for k in targets if 0 <= k < logits.numel()]
    if not keys:
        return None
    idx = torch.tensor(keys, dtype=torch.long, device=dev)
    y = torch.tensor([1.0 if targets[k] else 0.0 for k in keys], device=dev)
    return F.binary_cross_entropy_with_logits(logits[idx], y)


def _mse_loss(pred: torch.Tensor, targets: dict[int, float], dev) -> Optional[torch.Tensor]:
    if pred.numel() == 0 or not targets:
        return None
    keys = [k for k in targets if 0 <= k < pred.numel()]
    if not keys:
        return None
    idx = torch.tensor(keys, dtype=torch.long, device=dev)
    y = torch.tensor([float(targets[k]) for k in keys], device=dev)
    return F.mse_loss(pred[idx], y)
