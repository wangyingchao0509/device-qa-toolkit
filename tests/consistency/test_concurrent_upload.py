"""并发一致性：20 个线程同时上报同一条数据，只能入库一次。

为什么单独成一个模块：
- 它是"并发缺陷"的典型形态，顺序执行永远测不出来；
- 这类用例是硬件侧"多设备同时上报"场景的最小复现；
- 面试时能讲清"为什么测这个、失败说明什么"，比多写十条 CRUD 有价值。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from utils.data_factory import reading_payload
from utils.http_client import HttpClient

THREADS = 20


@pytest.mark.consistency
def test_concurrent_duplicate_upload_only_once(client: HttpClient, device: dict) -> None:
    payload = reading_payload(device["sn"], ts=1_760_000_900)

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        results = list(pool.map(lambda _: client.post("/api/readings", json=payload), range(THREADS)))

    created = [r for r in results if r.status_code == 201]
    duplicated = [r for r in results if r.status_code == 200]

    assert len(created) == 1, f"应只有 1 次真正创建，实际 {len(created)} 次"
    assert len(duplicated) == THREADS - 1, f"其余应全部判定为重复，实际 {len(duplicated)} 次"

    listed = client.get(f"/api/readings?sn={device['sn']}")
    assert len(listed.data) == 1, f"并发后库里应只有 1 条记录，实际 {len(listed.data)} 条"


@pytest.mark.consistency
def test_concurrent_distinct_readings_not_lost(client: HttpClient, device: dict) -> None:
    """并发写入不同时间戳时，不能丢数据（丢数据比重复更隐蔽）。"""

    def upload(i: int):
        return client.post("/api/readings", json=reading_payload(device["sn"], ts=1_760_001_000 + i))

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        results = list(pool.map(upload, range(THREADS)))

    assert all(r.status_code == 201 for r in results), "并发写入不同数据时不应出现失败"

    listed = client.get(f"/api/readings?sn={device['sn']}&limit=1000")
    assert len(listed.data) == THREADS, f"应有 {THREADS} 条，实际 {len(listed.data)} 条"
