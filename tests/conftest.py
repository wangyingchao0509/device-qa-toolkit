"""pytest 全局夹具。

分层设计（这是面试时值得讲的一点）：
    session 级：启动一次被测服务，全流程复用，避免每个用例重启
    function 级：每个用例前清空数据，保证用例之间零耦合，可单独运行、可乱序运行
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

from mock_server.app import create_server
from utils.data_factory import device_payload
from utils.http_client import HttpClient


@pytest.fixture(scope="session")
def server() -> Iterator[str]:
    """在随机空闲端口启动 Mock 服务，测完关闭。"""
    httpd = create_server(port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{httpd.server_port}"
    yield base_url
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def client(server: str) -> HttpClient:
    """每个用例独立客户端；用例开始前重置服务端数据。"""
    c = HttpClient(server)
    c.reset()
    return c


@pytest.fixture
def device(client: HttpClient) -> dict:
    """预置一台已注册设备，供需要设备上下文的用例使用。"""
    payload = device_payload()
    resp = client.post("/api/devices", json=payload)
    assert resp.status_code == 201, f"预置设备失败：{resp.text}"
    return resp.data
