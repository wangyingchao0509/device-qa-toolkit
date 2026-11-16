"""HTTP 客户端封装：统一超时、日志、断言入口。

测试代码里不直接写 requests.get(...)，好处是：
1. 超时、重试、基础 URL 只在一处改；
2. 出问题时日志里能看到完整的请求与响应，定位快。
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


class ApiResponse:
    def __init__(self, resp: requests.Response, elapsed_ms: float) -> None:
        self.status_code = resp.status_code
        self.headers = resp.headers
        self.elapsed_ms = elapsed_ms
        try:
            self.json: dict[str, Any] = resp.json()
        except ValueError:
            self.json = {}
        self.text = resp.text

    @property
    def code(self) -> str:
        return self.json.get("code", "")

    @property
    def data(self) -> Any:
        return self.json.get("data")

    def __repr__(self) -> str:  # 断言失败时打印这个
        return f"<ApiResponse {self.status_code} {self.elapsed_ms:.0f}ms {self.text[:200]}>"


class HttpClient:
    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> ApiResponse:
        url = f"{self.base_url}{path}"
        start = time.perf_counter()
        resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        api = ApiResponse(resp, elapsed)
        logger.info("%s %s -> %s (%.0fms)", method, path, resp.status_code, elapsed)
        return api

    def get(self, path: str, **kwargs: Any) -> ApiResponse:
        return self._request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> ApiResponse:
        return self._request("POST", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> ApiResponse:
        return self._request("DELETE", path, **kwargs)

    def reset(self) -> None:
        self.post("/api/admin/reset")
