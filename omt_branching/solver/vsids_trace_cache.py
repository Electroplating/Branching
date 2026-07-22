"""VSIDS 轨迹标签缓存。

同一 ``.smt2`` + 相同 ``VSIDSTraceConfig(stride, max_examples)`` 下轨迹确定性，故可按
实例落盘，避免 imitation 每次重跑观察-only 求解。布局::

    vsids_trace/<split>/<instance_id>.json

缓存存中间状态记录（assignment / trail / chosen_key / phase）；建 ``RankingExample`` 时仍需从
``.smt2`` 按各 assignment 建图，再映射到图内局部索引（建图远比重跑 VSIDS 轨迹便宜）。
``weight`` 仅在建样本时使用，不参与缓存指纹。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

VSIDS_TRACE_SUBDIR = "vsids_trace"


def vsids_trace_result_path(
    dataset_dir,
    instance_id: str,
    *,
    split: str,
) -> Path:
    return Path(dataset_dir) / VSIDS_TRACE_SUBDIR / split / f"{instance_id}.json"


def has_vsids_trace_result(dataset_dir, instance_id: str, *, split: str) -> bool:
    return vsids_trace_result_path(dataset_dir, instance_id, split=split).is_file()


def _config_fingerprint(stride: int, max_examples: int) -> dict:
    return {"stride": int(stride), "max_examples": int(max_examples)}


def save_vsids_trace_result(
    dataset_dir,
    instance_id: str,
    *,
    split: str,
    records: list,
    ref_conflicts: int = 0,
    info: dict | None = None,
    stride: int = 1,
    max_examples: int = 0,
) -> Path:
    """立刻写入 VSIDS 轨迹记录。"""
    path = vsids_trace_result_path(dataset_dir, instance_id, split=split)
    path.parent.mkdir(parents=True, exist_ok=True)
    ser_records = []
    for rec in records:
        if len(rec) == 4:
            assignment, trail, chosen_key, phase = rec
        else:
            assignment, chosen_key, phase = rec
            trail = list(assignment.keys())
        ser_records.append({
            "assignment": {str(k): bool(v) for k, v in assignment.items()},
            "trail": [str(k) for k in trail],
            "chosen_key": str(chosen_key),
            "phase": bool(phase),
        })
    payload = {
        "instance_id": instance_id,
        "split": split,
        "config": _config_fingerprint(stride, max_examples),
        "ref_conflicts": int(ref_conflicts),
        "info": info or {},
        "records": ser_records,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def load_vsids_trace_result(
    dataset_dir,
    instance_id: str,
    *,
    split: str,
    stride: int | None = None,
    max_examples: int | None = None,
) -> Optional[dict]:
    """加载缓存；若给定 config 且与落盘不一致则返回 ``None``（需重算）。"""
    path = vsids_trace_result_path(dataset_dir, instance_id, split=split)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    cfg = payload.get("config") or {}
    if stride is not None and int(cfg.get("stride", -1)) != int(stride):
        return None
    if max_examples is not None and int(cfg.get("max_examples", -1)) != int(max_examples):
        return None
    records = []
    for r in payload.get("records") or []:
        asg = {str(k): bool(v) for k, v in (r.get("assignment") or {}).items()}
        trail = r.get("trail")
        if trail is None:
            trail = list(asg.keys())
        else:
            trail = [str(k) for k in trail]
        records.append((asg, trail, str(r["chosen_key"]), bool(r["phase"])))
    return {
        "records": records,
        "ref_conflicts": int(payload.get("ref_conflicts") or 0),
        "info": payload.get("info") or {},
        "config": cfg,
    }


__all__ = [
    "VSIDS_TRACE_SUBDIR",
    "vsids_trace_result_path",
    "has_vsids_trace_result",
    "save_vsids_trace_result",
    "load_vsids_trace_result",
]
