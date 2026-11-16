"""采集数据上报：数值边界 + 幂等 + 设备存在性。

对应硬件场景：设备上报的传感器数据是"物理量"，边界要按规格书上
下限取值（温度 -40~85℃、电量 0~100%、心率 30~220 bpm）。
"""

from __future__ import annotations

import pytest

from utils.assertions import assert_status
from utils.data_factory import reading_payload
from utils.http_client import HttpClient


@pytest.mark.api
@pytest.mark.parametrize(
    "temp_c, expect_status",
    [
        (-40.0, 201),   # 规格下限
        (85.0, 201),    # 规格上限
        (-40.1, 400),   # 下限外
        (85.1, 400),    # 上限外
        (25.0, 201),    # 典型值
    ],
    ids=["temp_-40", "temp_85", "temp_-40.1", "temp_85.1", "temp_25"],
)
def test_temp_boundary(client: HttpClient, device: dict, temp_c: float, expect_status: int) -> None:
    resp = client.post("/api/readings", json=reading_payload(device["sn"], temp_c=temp_c))
    assert resp.status_code == expect_status, (
        f"温度 {temp_c} 期望 {expect_status}，实际 {resp.status_code}，响应 {resp.text[:200]}"
    )


@pytest.mark.api
@pytest.mark.parametrize(
    "battery_pct, expect_status",
    [(0, 201), (100, 201), (-1, 400), (101, 400)],
    ids=["battery_0", "battery_100", "battery_-1", "battery_101"],
)
def test_battery_boundary(client: HttpClient, device: dict, battery_pct: int, expect_status: int) -> None:
    resp = client.post("/api/readings", json=reading_payload(device["sn"], battery_pct=battery_pct))
    assert resp.status_code == expect_status


@pytest.mark.api
@pytest.mark.parametrize(
    "heart_rate, expect_status",
    [(30, 201), (220, 201), (29, 400), (221, 400), (None, 201)],
    ids=["hr_30", "hr_220", "hr_29", "hr_221", "hr_null"],
)
def test_heart_rate_boundary(client: HttpClient, device: dict, heart_rate, expect_status: int) -> None:
    """心率允许为 null —— 未佩戴时不应上报 0 或伪造值。"""
    resp = client.post("/api/readings", json=reading_payload(device["sn"], heart_rate=heart_rate))
    assert resp.status_code == expect_status


@pytest.mark.api
def test_reading_for_unregistered_device(client: HttpClient) -> None:
    resp = client.post("/api/readings", json=reading_payload("ZZ99999999"))
    assert_status(resp, 404)
    assert "未注册" in resp.text


@pytest.mark.api
def test_duplicate_reading_is_idempotent(client: HttpClient, device: dict) -> None:
    """同一设备同一时间戳重复上报，应去重而不是累加。

    这是真实项目里的高频缺陷：网络重试导致数据重复入库，
    最终把统计口径全部带偏。
    """
    payload = reading_payload(device["sn"], ts=1_760_000_100)
    first = client.post("/api/readings", json=payload)
    second = client.post("/api/readings", json=payload)

    assert_status(first, 201)
    assert_status(second, 200), "重复上报应返回 200 且标记 duplicated"
    assert second.json["duplicated"] is True

    listed = client.get(f"/api/readings?sn={device['sn']}")
    assert len(listed.data) == 1, f"去重后应只有 1 条，实际 {len(listed.data)} 条"


@pytest.mark.api
def test_summary_statistics(client: HttpClient, device: dict) -> None:
    for i, temp in enumerate([30.0, 40.0, 50.0]):
        client.post("/api/readings", json=reading_payload(device["sn"], ts=1_760_000_200 + i, temp_c=temp))

    resp = client.get(f"/api/devices/{device['sn']}/summary")
    assert_status(resp, 200)
    assert resp.data["count"] == 3
    assert resp.data["avg_temp_c"] == pytest.approx(40.0, abs=0.01)
    assert resp.data["max_temp_c"] == 50.0
