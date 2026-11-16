"""Locust 负载脚本：python -m locust -f perf/locustfile.py --host http://127.0.0.1:8000

与 run_load_test.py 的区别：
- locust 自带 Web UI、分布式与实时曲线，适合长时间观察；
- run_load_test.py 输出与仓库统一风格的 Markdown 报告，适合贴进 CI。

两者场景配比保持一致，便于交叉验证结论。
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task


class DeviceUser(HttpUser):
    """模拟一台设备：周期性上报数据，偶尔查询。"""

    wait_time = between(0.05, 0.2)  # 思考时间，不加会被放大成数倍压力

    def on_start(self) -> None:
        # 每台虚拟设备注册唯一 SN，避免 409 冲突导致压测数据失真
        self.sn = "LT" + "".join(random.choices("0123456789", k=8))
        self.client.post(
            "/api/devices",
            json={"sn": self.sn, "name": f"locust-{self.sn}", "model": "WATCH-S1"},
            name="POST /api/devices (setup)",
        )

    @task(55)
    def report_reading(self) -> None:
        payload = {
            "sn": self.sn,
            "ts": random.randint(1_760_000_000, 1_800_000_000),
            "temp_c": round(random.gauss(36.5, 2.0), 2),
            "battery_pct": random.randint(10, 100),
            "heart_rate": random.randint(55, 130),
        }
        self.client.post("/api/readings", json=payload, name="POST /api/readings")

    @task(30)
    def query_readings(self) -> None:
        self.client.get(
            "/api/readings",
            params={"sn": self.sn, "limit": 50},
            name="GET /api/readings",
        )

    @task(15)
    def query_summary(self) -> None:
        self.client.get(f"/api/devices/{self.sn}/summary", name="GET /api/devices/[sn]/summary")
