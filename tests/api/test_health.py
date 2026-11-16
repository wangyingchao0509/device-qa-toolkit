"""冒烟层：服务是否活着、依赖是否就绪。

这一层对应真实项目里的「发版后冒烟」，P0，必须最先跑、最先绿。
"""

from __future__ import annotations

import pytest

from utils.assertions import assert_field, assert_ok, assert_status
from utils.http_client import HttpClient


@pytest.mark.smoke
def test_health_returns_ok(client: HttpClient) -> None:
    resp = client.get("/health")
    assert_status(resp, 200)
    assert_field(resp.json, "status", "ok")
    assert "ts" in resp.json, "健康检查应返回时间戳，便于排查服务是否卡死"


@pytest.mark.smoke
def test_device_list_is_empty_after_reset(client: HttpClient) -> None:
    resp = client.get("/api/devices")
    assert_ok(resp)
    assert resp.data == [], f"重置后设备列表应为空，实际 {resp.data}"
