"""测试数据构造：保证每条用例拿到的数据互不干扰，且可复现。

为什么需要它：
- 手写 sn 容易重复，跑第二遍就撞 409；
- 用当前时间做种子，失败时能根据日志里的 seed 复现问题。
"""

from __future__ import annotations

import random
import string
from typing import Any


def random_sn(seed: int | None = None) -> str:
    rng = random.Random(seed)
    prefix = "".join(rng.choices(string.ascii_uppercase, k=2))
    digits = "".join(rng.choices(string.digits, k=8))
    return f"{prefix}{digits}"


def device_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "sn": random_sn(),
        "name": "测试手表",
        "model": "WATCH-S1",
        "fw_version": "1.2.0",
    }
    payload.update(overrides)
    return payload


def reading_payload(sn: str, **overrides: Any) -> dict[str, Any]:
    payload = {
        "sn": sn,
        "ts": 1_760_000_000,
        "temp_c": 36.5,
        "battery_pct": 88,
        "heart_rate": 72,
    }
    payload.update(overrides)
    return payload
