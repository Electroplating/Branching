"""z3 二进制求解结果缓存。

同一 ``.smt2`` 上 ``solve_binary`` 结果确定性，故按实例落盘后可反复复用（评测 / RL reward）。
布局（相对数据集根目录）::

    binary/<split>/<instance_id>.json

并行求解时每实例独立文件，完成后立即写入，无共享锁。
"""

from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

BINARY_SUBDIR = "binary"


def binary_result_path(
    dataset_dir,
    instance_id: str,
    *,
    split: str,
) -> Path:
    """返回某实例 binary 结果 JSON 路径。"""
    return Path(dataset_dir) / BINARY_SUBDIR / split / f"{instance_id}.json"


def has_binary_result(dataset_dir, instance_id: str, *, split: str) -> bool:
    return binary_result_path(dataset_dir, instance_id, split=split).is_file()


def _json_default(v: Any):
    if isinstance(v, Fraction):
        return str(v)
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def serialize_binary_result(result: dict) -> dict:
    """把 ``solve_binary`` 返回值转为 JSON 可序列化 dict（保留全字段）。"""
    out: dict = {}
    for k, v in result.items():
        if k == "z3_stats" and isinstance(v, dict):
            out[k] = {sk: _json_default(sv) for sk, sv in v.items()}
        else:
            out[k] = _json_default(v)
    return out


def _parse_value(raw) -> Any:
    if raw is None or isinstance(raw, (int, float, Fraction)):
        return raw
    if isinstance(raw, str):
        try:
            return Fraction(raw)
        except (ValueError, ZeroDivisionError):
            return raw
    return raw


def deserialize_binary_result(payload: dict) -> dict:
    """读回落盘结果：把 ``value`` 尽量还原为 ``Fraction``。"""
    out = dict(payload)
    if "value" in out:
        out["value"] = _parse_value(out["value"])
    return out


def save_binary_result(
    dataset_dir,
    instance_id: str,
    result: dict,
    *,
    split: str,
) -> Path:
    """立刻把结果写入磁盘（先写临时文件再 rename，避免半截 JSON）。"""
    path = binary_result_path(dataset_dir, instance_id, split=split)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_binary_result(result)
    payload.setdefault("instance_id", instance_id)
    payload.setdefault("split", split)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def load_binary_result(
    dataset_dir,
    instance_id: str,
    *,
    split: str,
) -> Optional[dict]:
    """加载单实例结果；不存在返回 ``None``。"""
    path = binary_result_path(dataset_dir, instance_id, split=split)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return deserialize_binary_result(json.load(f))


def binary_rlimit(
    dataset_dir,
    instance_id: str,
    *,
    split: str,
) -> Optional[int]:
    """RL reward 用：返回缓存的 ``rlimit``（无结果或超时则为 ``None``）。"""
    res = load_binary_result(dataset_dir, instance_id, split=split)
    if res is None:
        return None
    rl = res.get("rlimit")
    return int(rl) if rl is not None else None


def binary_value(
    dataset_dir,
    instance_id: str,
    *,
    split: str,
):
    """RL / match 用：返回缓存的最优 ``value``（``Fraction`` 或 ``None``）。"""
    res = load_binary_result(dataset_dir, instance_id, split=split)
    if res is None:
        return None
    return res.get("value")


def load_binary_results(
    dataset_dir,
    *,
    split: str | None = None,
) -> dict[str, dict]:
    """批量加载 ``binary/`` 下结果，键为 ``instance_id``。

    ``split`` 为 ``None`` 时扫描所有划分；同 id 后写覆盖先写（通常各 split id 不冲突）。
    """
    root = Path(dataset_dir) / BINARY_SUBDIR
    if not root.is_dir():
        return {}
    splits = [split] if split is not None else sorted(
        p.name for p in root.iterdir() if p.is_dir()
    )
    out: dict[str, dict] = {}
    for sp in splits:
        sp_dir = root / sp
        if not sp_dir.is_dir():
            continue
        for path in sorted(sp_dir.glob("*.json")):
            with open(path, encoding="utf-8") as f:
                payload = deserialize_binary_result(json.load(f))
            iid = payload.get("instance_id") or path.stem
            out[iid] = payload
    return out


def missing_binary_ids(
    dataset_dir,
    entries: list[dict],
    *,
    split: str,
) -> list[str]:
    """给出 manifest 条目中尚未有 binary 结果的 ``instance_id`` 列表。"""
    missing: list[str] = []
    for e in entries:
        iid = e["instance_id"]
        if not has_binary_result(dataset_dir, iid, split=split):
            missing.append(iid)
    return missing


__all__ = [
    "BINARY_SUBDIR",
    "binary_result_path",
    "has_binary_result",
    "serialize_binary_result",
    "deserialize_binary_result",
    "save_binary_result",
    "load_binary_result",
    "binary_rlimit",
    "binary_value",
    "load_binary_results",
    "missing_binary_ids",
]
