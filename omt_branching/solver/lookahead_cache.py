"""Look-ahead 标签缓存。

同一 ``.smt2`` + 相同 ``LookaheadConfig`` 下 ``lookahead_scores`` 确定性，故可按实例落盘，
避免 imitation 每次重跑 z3 consequences。布局::

    lookahead/<split>/<instance_id>.json

缓存存 atom_key -> score/phase（字符串键）；建 ``RankingExample`` 时仍需从 .smt2 建图，
再把缓存映射到图内局部索引（建图远比 look-ahead 便宜）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

LOOKAHEAD_SUBDIR = "lookahead"


def lookahead_result_path(
    dataset_dir,
    instance_id: str,
    *,
    split: str,
) -> Path:
    return Path(dataset_dir) / LOOKAHEAD_SUBDIR / split / f"{instance_id}.json"


def has_lookahead_result(dataset_dir, instance_id: str, *, split: str) -> bool:
    return lookahead_result_path(dataset_dir, instance_id, split=split).is_file()


def _config_fingerprint(max_atoms: int, eps: float) -> dict:
    return {"max_atoms": int(max_atoms), "eps": float(eps)}


def save_lookahead_result(
    dataset_dir,
    instance_id: str,
    *,
    split: str,
    scores: dict[str, float],
    phases: dict[str, bool],
    max_atoms: int,
    eps: float,
) -> Path:
    """立刻写入 look-ahead 标签（原子键 -> 分数/相位）。"""
    path = lookahead_result_path(dataset_dir, instance_id, split=split)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "instance_id": instance_id,
        "split": split,
        "config": _config_fingerprint(max_atoms, eps),
        "scores": {str(k): float(v) for k, v in scores.items()},
        "phases": {str(k): bool(v) for k, v in phases.items()},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def load_lookahead_result(
    dataset_dir,
    instance_id: str,
    *,
    split: str,
    max_atoms: int | None = None,
    eps: float | None = None,
) -> Optional[dict]:
    """加载缓存；若给定 config 且与落盘不一致则返回 ``None``（需重算）。"""
    path = lookahead_result_path(dataset_dir, instance_id, split=split)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    cfg = payload.get("config") or {}
    if max_atoms is not None and int(cfg.get("max_atoms", -1)) != int(max_atoms):
        return None
    if eps is not None and abs(float(cfg.get("eps", -1.0)) - float(eps)) > 1e-15:
        return None
    return {
        "scores": {str(k): float(v) for k, v in (payload.get("scores") or {}).items()},
        "phases": {str(k): bool(v) for k, v in (payload.get("phases") or {}).items()},
        "config": cfg,
    }


__all__ = [
    "LOOKAHEAD_SUBDIR",
    "lookahead_result_path",
    "has_lookahead_result",
    "save_lookahead_result",
    "load_lookahead_result",
]
