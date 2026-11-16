"""被测服务（本地 Mock）：一个极简的设备上报接口。

设计意图：
1. 让整个仓库 `git clone` 后无需任何外部依赖即可运行 —— 面试官最烦"跑不起来"。
2. 覆盖真实接口测试需要的典型校验：字段格式、数值边界、幂等去重、并发安全。
3. 用标准库实现，不引入 Web 框架，避免评审时被质疑"重点变成了搭服务"。

接口清单
--------
GET    /health                          健康检查
POST   /api/devices                     创建设备
GET    /api/devices                     设备列表
GET    /api/devices/{sn}                设备详情
DELETE /api/devices/{sn}                删除设备
POST   /api/readings                    上报一条采集数据
GET    /api/readings?sn=XX&limit=N      查询采集数据
GET    /api/devices/{sn}/summary        设备汇总统计
POST   /api/admin/reset                 清空数据（仅供测试隔离使用）
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

SN_PATTERN = re.compile(r"^[A-Z]{2}\d{8}$")
NAME_MAX_LEN = 32
TEMP_RANGE = (-40.0, 85.0)
BATTERY_RANGE = (0, 100)
HEART_RATE_RANGE = (30, 220)


class Store:
    """内存存储，加锁保证并发安全。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.devices: dict[str, dict[str, Any]] = {}
        self.readings: dict[tuple[str, int], dict[str, Any]] = {}

    def reset(self) -> None:
        with self._lock:
            self.devices.clear()
            self.readings.clear()

    def add_device(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.devices[payload["sn"]] = payload
        return payload

    def get_device(self, sn: str) -> dict[str, Any] | None:
        return self.devices.get(sn)

    def delete_device(self, sn: str) -> bool:
        with self._lock:
            existed = self.devices.pop(sn, None) is not None
            for key in [k for k in self.readings if k[0] == sn]:
                self.readings.pop(key, None)
        return existed

    def add_reading(self, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """按 (sn, ts) 幂等写入，返回 (记录, 是否为重复上报)。"""
        key = (payload["sn"], payload["ts"])
        with self._lock:
            if key in self.readings:
                return self.readings[key], True
            self.readings[key] = payload
            return payload, False

    def list_readings(self, sn: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = [v for k, v in sorted(self.readings.items()) if k[0] == sn]
        return rows[:limit]

    def summary(self, sn: str) -> dict[str, Any]:
        rows = self.list_readings(sn, limit=100000)
        if not rows:
            return {"sn": sn, "count": 0, "avg_temp_c": None, "max_temp_c": None}
        temps = [r["temp_c"] for r in rows]
        return {
            "sn": sn,
            "count": len(rows),
            "avg_temp_c": round(sum(temps) / len(temps), 2),
            "max_temp_c": max(temps),
        }


STORE = Store()


def _error(message: str, code: str = "INVALID_PARAM") -> dict[str, Any]:
    return {"code": code, "message": message}


def validate_device(body: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    sn = body.get("sn", "")
    name = body.get("name", "")
    if not SN_PATTERN.match(str(sn)):
        errors.append("sn 必须为 2 位大写字母 + 8 位数字，例如 AB12345678")
    if not isinstance(name, str) or not (1 <= len(name) <= NAME_MAX_LEN):
        errors.append(f"name 长度必须在 1~{NAME_MAX_LEN} 之间")
    if not body.get("model"):
        errors.append("model 不能为空")
    return errors


def validate_reading(body: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not SN_PATTERN.match(str(body.get("sn", ""))):
        errors.append("sn 格式非法")
    ts = body.get("ts")
    if not isinstance(ts, int) or ts <= 0:
        errors.append("ts 必须是正整数时间戳（秒）")
    temp = body.get("temp_c")
    if not isinstance(temp, (int, float)) or not (TEMP_RANGE[0] <= temp <= TEMP_RANGE[1]):
        errors.append(f"temp_c 必须在 {TEMP_RANGE[0]}~{TEMP_RANGE[1]} 之间")
    battery = body.get("battery_pct")
    if not isinstance(battery, int) or not (BATTERY_RANGE[0] <= battery <= BATTERY_RANGE[1]):
        errors.append("battery_pct 必须是 0~100 的整数")
    hr = body.get("heart_rate")
    if hr is not None and (not isinstance(hr, int) or not (HEART_RATE_RANGE[0] <= hr <= HEART_RATE_RANGE[1])):
        errors.append(f"heart_rate 必须在 {HEART_RATE_RANGE[0]}~{HEART_RATE_RANGE[1]} 之间，或为 null")
    return errors


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # 静默，避免污染测试输出
        return

    # ---------- 基础工具 ----------
    def _send(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    # ---------- 路由 ----------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path, query = parsed.path, parse_qs(parsed.query)

        if path == "/health":
            self._send(200, {"status": "ok", "ts": int(time.time())})
            return

        if path == "/api/devices":
            self._send(200, {"code": "OK", "data": list(STORE.devices.values())})
            return

        m = re.fullmatch(r"/api/devices/([A-Za-z0-9]+)", path)
        if m:
            device = STORE.get_device(m.group(1))
            if device is None:
                self._send(404, _error("设备不存在", "NOT_FOUND"))
            else:
                self._send(200, {"code": "OK", "data": device})
            return

        m = re.fullmatch(r"/api/devices/([A-Za-z0-9]+)/summary", path)
        if m:
            self._send(200, {"code": "OK", "data": STORE.summary(m.group(1))})
            return

        if path == "/api/readings":
            sn = (query.get("sn") or [""])[0]
            limit = int((query.get("limit") or ["100"])[0])
            self._send(200, {"code": "OK", "data": STORE.list_readings(sn, limit)})
            return

        self._send(404, _error("接口不存在", "NOT_FOUND"))

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        body = self._body()

        if path == "/api/admin/reset":
            STORE.reset()
            self._send(200, {"code": "OK"})
            return

        if path == "/api/devices":
            errors = validate_device(body)
            if errors:
                self._send(400, {**_error("; ".join(errors)), "errors": errors})
                return
            if STORE.get_device(body["sn"]):
                self._send(409, _error("设备已存在", "CONFLICT"))
                return
            device = {
                "sn": body["sn"],
                "name": body["name"],
                "model": body["model"],
                "fw_version": body.get("fw_version", "1.0.0"),
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self._send(201, {"code": "OK", "data": STORE.add_device(device)})
            return

        if path == "/api/readings":
            errors = validate_reading(body)
            if errors:
                self._send(400, {**_error("; ".join(errors)), "errors": errors})
                return
            if STORE.get_device(body["sn"]) is None:
                self._send(404, _error("设备未注册", "NOT_FOUND"))
                return
            row, duplicated = STORE.add_reading(body)
            self._send(200 if duplicated else 201, {"code": "OK", "data": row, "duplicated": duplicated})
            return

        self._send(404, _error("接口不存在", "NOT_FOUND"))

    def do_DELETE(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        m = re.fullmatch(r"/api/devices/([A-Za-z0-9]+)", path)
        if m:
            if STORE.delete_device(m.group(1)):
                self._send(200, {"code": "OK"})
            else:
                self._send(404, _error("设备不存在", "NOT_FOUND"))
            return
        self._send(404, _error("接口不存在", "NOT_FOUND"))


def create_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """创建服务实例；port=0 时由系统分配空闲端口。"""
    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    server = create_server(port=8000)
    print(f"Mock server listening on http://127.0.0.1:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
