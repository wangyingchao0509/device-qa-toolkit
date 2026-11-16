"""接口契约（Schema）校验：防止字段缺失、类型漂移。

真实价值：上游改了字段类型（比如 battery_pct 从 int 变 string），
业务断言可能仍然通过，但下游（报表、App）会静默出错。
这一层专门守接口契约。
"""

from __future__ import annotations

import pytest

from utils.data_factory import reading_payload
from utils.http_client import HttpClient

REQUIRED_READING_FIELDS = {
    "sn": str,
    "ts": int,
    "temp_c": (int, float),
    "battery_pct": int,
}


@pytest.mark.data
def test_reading_schema_types(client: HttpClient, device: dict) -> None:
    client.post("/api/readings", json=reading_payload(device["sn"], ts=1_760_002_000))
    resp = client.get(f"/api/readings?sn={device['sn']}")

    assert resp.data, "应至少返回一条数据"
    row = resp.data[0]

    for field, expected_type in REQUIRED_READING_FIELDS.items():
        assert field in row, f"响应缺少必填字段 {field}，实际字段 {list(row)}"
        assert isinstance(row[field], expected_type), (
            f"字段 {field} 类型应为 {expected_type}，实际 {type(row[field])}"
        )


@pytest.mark.data
def test_heart_rate_may_be_null(client: HttpClient, device: dict) -> None:
    """心率可以为 null（未佩戴），但不能是 0 或字符串。"""
    client.post("/api/readings", json=reading_payload(device["sn"], ts=1_760_002_100, heart_rate=None))
    row = client.get(f"/api/readings?sn={device['sn']}").data[0]
    assert row["heart_rate"] is None


@pytest.mark.data
def test_readings_sorted_by_timestamp(client: HttpClient, device: dict) -> None:
    """返回顺序必须按时间升序，否则前端画曲线会乱。"""
    timestamps = [1_760_003_000 + i for i in (3, 1, 2)]
    for ts in timestamps:
        client.post("/api/readings", json=reading_payload(device["sn"], ts=ts))

    rows = client.get(f"/api/readings?sn={device['sn']}").data
    got = [r["ts"] for r in rows]
    assert got == sorted(got), f"应按时间升序返回，实际 {got}"
