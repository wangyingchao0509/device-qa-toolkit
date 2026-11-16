"""设备管理接口：正向流程 + 边界值 + 异常分支。

用例设计方法体现：
- 等价类 + 边界值：name 长度 0/1/32/33，sn 格式合法/非法
- 场景法：创建→查询→删除→再查询（状态流转）
- 错误猜测：重复创建、删除不存在的设备
"""

from __future__ import annotations

import pytest

from utils.assertions import assert_code, assert_error_contains, assert_ok, assert_status
from utils.data_factory import device_payload
from utils.http_client import HttpClient


# ---------------- 正向流程 ----------------

@pytest.mark.api
def test_create_device_success(client: HttpClient) -> None:
    payload = device_payload(name="测试手表-A1")
    resp = client.post("/api/devices", json=payload)

    assert_status(resp, 201)
    assert_code(resp, "OK")
    assert resp.data["sn"] == payload["sn"]
    assert resp.data["name"] == payload["name"]
    assert resp.data["created_at"], "创建时间不能为空，后续做批次追溯要用"


@pytest.mark.api
def test_get_device_detail(client: HttpClient, device: dict) -> None:
    resp = client.get(f"/api/devices/{device['sn']}")
    assert_ok(resp)
    assert resp.data["sn"] == device["sn"]
    assert resp.data["model"] == device["model"]


@pytest.mark.api
def test_device_lifecycle_create_query_delete(client: HttpClient) -> None:
    """完整状态流转：创建后能查到，删除后查不到。"""
    payload = device_payload()
    assert_status(client.post("/api/devices", json=payload), 201)

    assert_ok(client.get(f"/api/devices/{payload['sn']}"))
    assert_ok(client.delete(f"/api/devices/{payload['sn']}"))

    gone = client.get(f"/api/devices/{payload['sn']}")
    assert_status(gone, 404)
    assert_code(gone, "NOT_FOUND")


# ---------------- 边界值：name 长度 ----------------

@pytest.mark.api
@pytest.mark.parametrize(
    "name, expect_status",
    [
        ("A", 201),                 # 下边界内
        ("A" * 32, 201),            # 上边界内
        ("", 400),                  # 下边界外（空串）
        ("A" * 33, 400),            # 上边界外
    ],
    ids=["name_len_1", "name_len_32", "name_len_0", "name_len_33"],
)
def test_create_device_name_boundary(client: HttpClient, name: str, expect_status: int) -> None:
    payload = device_payload(name=name)
    resp = client.post("/api/devices", json=payload)
    assert resp.status_code == expect_status, (
        f"name 长度 {len(name)} 期望 {expect_status}，实际 {resp.status_code}，响应 {resp.text[:200]}"
    )


# ---------------- 异常分支：sn 格式 ----------------

@pytest.mark.api
@pytest.mark.parametrize(
    "sn",
    [
        "ab12345678",    # 小写字母
        "AB1234567",     # 位数不足
        "AB123456789",   # 位数超出
        "A123456789",    # 前缀不足 2 位
        "",              # 空值
    ],
    ids=["lowercase", "short", "long", "prefix_1", "empty"],
)
def test_create_device_invalid_sn(client: HttpClient, sn: str) -> None:
    resp = client.post("/api/devices", json=device_payload(sn=sn))
    assert_status(resp, 400)
    assert_error_contains(resp, "sn")


# ---------------- 异常分支：重复与不存在 ----------------

@pytest.mark.api
def test_create_duplicate_device_conflict(client: HttpClient, device: dict) -> None:
    resp = client.post("/api/devices", json=device_payload(sn=device["sn"]))
    assert_status(resp, 409)
    assert_code(resp, "CONFLICT")


@pytest.mark.api
def test_delete_unknown_device(client: HttpClient) -> None:
    resp = client.delete("/api/devices/ZZ99999999")
    assert_status(resp, 404)


@pytest.mark.api
def test_missing_required_field(client: HttpClient) -> None:
    resp = client.post("/api/devices", json={"sn": "AB12345678"})
    assert_status(resp, 400)
    assert_error_contains(resp, "name")
