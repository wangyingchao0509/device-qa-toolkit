"""断言封装：让失败信息自带上下文。

原则：断言失败时，日志里必须能直接看出「期望什么、实际什么、哪条数据」。
否则一条参数化用例挂了，你要靠猜。
"""

from __future__ import annotations

from typing import Any

from utils.http_client import ApiResponse
from utils.data_factory import device_payload


def assert_status(resp: ApiResponse, expected: int) -> None:
    assert resp.status_code == expected, (
        f"状态码不符：期望 {expected}，实际 {resp.status_code}，响应 {resp.text[:300]}"
    )


def assert_code(resp: ApiResponse, expected: str) -> None:
    assert resp.code == expected, f"业务码不符：期望 {expected}，实际 {resp.code}，响应 {resp.text[:300]}"


def assert_ok(resp: ApiResponse) -> None:
    assert_status(resp, 200)
    assert_code(resp, "OK")


def assert_error_contains(resp: ApiResponse, keyword: str) -> None:
    """校验错误信息里包含关键字 —— 比只校验状态码更能发现"错误提示写错"的问题。"""
    assert keyword in resp.text, f"错误信息中未包含「{keyword}」，实际响应 {resp.text[:300]}"


def assert_field(data: dict[str, Any], field: str, expected: Any) -> None:
    assert field in data, f"响应缺少字段 {field}，实际字段 {list(data)}"
    assert data[field] == expected, f"字段 {field} 不符：期望 {expected}，实际 {data[field]}"


__all__ = [
    "assert_status",
    "assert_code",
    "assert_ok",
    "assert_error_contains",
    "assert_field",
    "device_payload",
]
