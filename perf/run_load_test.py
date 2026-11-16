#!/usr/bin/env python3
"""轻量负载测试执行器：自带被测服务、自动出指标与图表。

为什么不用现成工具就够了：
- JMeter / locust 出的是工具视角的统计，面试官更想看你**怎么定义场景与判据**；
- 这个脚本把「场景配比 → 阶梯加压 → 指标计算 → 报告输出」串成一条链路，
  和仓库里的功能测试、数据分析共用一套报告风格。

用法：
    python perf/run_load_test.py --users 20 --duration 10
    python perf/run_load_test.py --users 50 --duration 30 --think-time 0.1 --report-dir reports
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mock_server.app import create_server  # noqa: E402

# 场景配比：贴近真实设备集群的行为分布
SCENARIOS = [
    # (名称, 方法, 路径构造, 权重, 说明)
    ("上报数据", "POST", "/api/readings", 55),
    ("查询数据", "GET", "/api/readings", 30),
    ("查询汇总", "GET", "/api/devices/{sn}/summary", 15),
]


@dataclass
class Sample:
    t: float
    latency_ms: float
    ok: bool
    scenario: str


@dataclass
class Stats:
    samples: list[Sample] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, s: Sample) -> None:
        with self.lock:
            self.samples.append(s)


def seed_devices(base_url: str, count: int) -> list[str]:
    """压测前批量注册设备，返回 SN 列表供参数化使用。"""
    sns = []
    for i in range(count):
        sn = f"PF{10000000 + i:08d}"
        requests.post(
            f"{base_url}/api/devices",
            json={"sn": sn, "name": f"压测设备{i}", "model": "WATCH-S1"},
            timeout=5,
        )
        sns.append(sn)
    return sns


def worker(
    base_url: str,
    sns: list[str],
    stats: Stats,
    start_at: float,
    stop_at: float,
    think_time: float,
    ts_base: int,
    idx: int,
) -> None:
    """单个虚拟用户的循环：按权重挑场景 → 发请求 → 记录延迟 → 思考时间。"""
    scen_list = [(method, path, weight, name) for name, method, path, weight in SCENARIOS]
    total_w = sum(item[2] for item in scen_list)
    rng = np.random.default_rng(idx)

    session = requests.Session()
    seq = 0

    # 阶梯加压：把虚拟用户的开始时间铺开在加压窗口内
    time.sleep(max(0.0, start_at - time.time()))

    while time.time() < stop_at:
        pick = float(rng.integers(0, total_w))
        acc = 0.0
        method, path, name = "GET", "/api/readings", "查询数据"
        for m, p, w, scen in scen_list:
            acc += w
            if pick < acc:
                method, path, name = m, p, scen
                break

        sn = sns[int(rng.integers(0, len(sns)))]
        path = path.replace("{sn}", sn)
        payload = None
        if method == "POST":
            seq += 1
            payload = {
                "sn": sn,
                "ts": ts_base + idx * 1_000_000 + seq,
                "temp_c": 36.0 + float(rng.normal(0, 2)),
                "battery_pct": int(rng.integers(20, 100)),
                "heart_rate": int(rng.integers(60, 120)),
            }

        t0 = time.perf_counter()
        ok = True
        try:
            if method == "POST":
                r = session.post(f"{base_url}{path}", json=payload, timeout=5)
            else:
                r = session.get(f"{base_url}{path}", params={"sn": sn, "limit": 50}, timeout=5)
            ok = r.status_code in (200, 201)
        except requests.RequestException:
            ok = False
        latency = (time.perf_counter() - t0) * 1000

        stats.add(Sample(t=time.time(), latency_ms=latency, ok=ok, scenario=name))
        if think_time > 0:
            time.sleep(think_time)


def percentile(values, p: float) -> float:
    return float(np.percentile(values, p)) if len(values) else 0.0


def summarize(stats: Stats, start: float, end: float) -> dict:
    samples = sorted(stats.samples, key=lambda s: s.t)
    total = len(samples)
    duration = max(end - start, 1e-6)
    ok_samples = [s for s in samples if s.ok]
    latencies = [s.latency_ms for s in ok_samples]

    by_scenario = {}
    for sc in {s.scenario for s in samples}:
        items = [s for s in samples if s.scenario == sc]
        lats = [s.latency_ms for s in items if s.ok]
        by_scenario[sc] = {
            "count": len(items),
            "error_rate_pct": round((len(items) - len(lats)) / len(items) * 100, 2) if items else 0.0,
            "avg_ms": round(float(np.mean(lats)), 1) if lats else 0.0,
            "p95_ms": round(percentile(lats, 95), 1),
        }

    return {
        "duration_s": round(duration, 1),
        "requests": total,
        "tps": round(total / duration, 2),
        "error_rate_pct": round((total - len(ok_samples)) / total * 100, 2) if total else 0.0,
        "avg_ms": round(float(np.mean(latencies)), 1) if latencies else 0.0,
        "p50_ms": round(percentile(latencies, 50), 1),
        "p95_ms": round(percentile(latencies, 95), 1),
        "p99_ms": round(percentile(latencies, 99), 1),
        "max_ms": round(max(latencies), 1) if latencies else 0.0,
        "scenarios": by_scenario,
        "_samples": samples,
        "_start": start,
    }


def plot_series(samples: list[Sample], out_path: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3})

    if not samples:
        return ""

    t0 = samples[0].t
    buckets: dict[int, list[Sample]] = {}
    for s in samples:
        buckets.setdefault(int(s.t - t0), []).append(s)

    secs = sorted(buckets)
    tps = [len(buckets[t]) for t in secs]
    p95 = [percentile([s.latency_ms for s in buckets[t] if s.ok], 95) for t in secs]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 4.2), sharex=True)

    ax1.plot(secs, tps, color="#1f4e79", linewidth=1.6)
    ax1.set_ylabel("TPS")
    ax1.set_title("Throughput and latency over time")

    ax2.plot(secs, p95, color="#c0392b", linewidth=1.6)
    ax2.set_ylabel("P95 latency (ms)")
    ax2.set_xlabel("Elapsed (s)")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def write_report(result: dict, args, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "perf_report.md")

    rows = []
    for name, m in sorted(result["scenarios"].items(), key=lambda kv: -kv[1]["count"]):
        rows.append(
            f"| {name} | {m['count']} | {m['avg_ms']} | {m['p95_ms']} | {m['error_rate_pct']}% |"
        )

    lines = [
        "# 负载测试报告",
        "",
        "## 一、测试条件",
        "",
        f"- 虚拟用户：**{args.users}**，加压窗口 {args.ramp} 秒，持续 {result['duration_s']} 秒",
        f"- 思考时间：{args.think_time} 秒（不加思考时间等于把并发压力放大数倍，结论会失真）",
        "- 场景配比：上报数据 55% / 查询数据 30% / 查询汇总 15%",
        "- 被测服务：仓库内置 Mock 服务（内存实现，非真实后端，瓶颈可能先出现在被测端）",
        "",
        "## 二、总体结果",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 总请求数 | {result['requests']} |",
        f"| 平均 TPS | {result['tps']} |",
        f"| 平均响应时间 | {result['avg_ms']} ms |",
        f"| P50 / P95 / P99 | {result['p50_ms']} / {result['p95_ms']} / {result['p99_ms']} ms |",
        f"| 最大响应时间 | {result['max_ms']} ms |",
        f"| 错误率 | {result['error_rate_pct']}% |",
        "",
        "## 三、分场景",
        "",
        "| 场景 | 请求数 | 平均 RT | P95 RT | 错误率 |",
        "|---|---|---|---|---|",
        *rows,
        "",
        "## 四、曲线",
        "",
        "![throughput](perf_throughput.png)",
        "",
        "## 五、判据与结论",
        "",
        "| 判据 | 阈值 | 实际 | 结论 |",
        "|---|---|---|---|",
        f"| 错误率 | ≤ 0.1% | {result['error_rate_pct']}% | "
        f"{'PASS' if result['error_rate_pct'] <= 0.1 else 'FAIL'} |",
        f"| P95 响应时间 | ≤ 500 ms | {result['p95_ms']} ms | "
        f"{'PASS' if result['p95_ms'] <= 500 else 'FAIL'} |",
        "",
        "> 说明：本仓库的 Mock 服务为单进程内存实现，CPU 与 GIL 是主要瓶颈。",
        "> 该结果用于验证压测脚本与指标口径，不代表真实后端容量。",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="设备上报接口负载测试")
    parser.add_argument("--users", type=int, default=20, help="虚拟用户数")
    parser.add_argument("--duration", type=int, default=10, help="持续时间（秒）")
    parser.add_argument("--ramp", type=int, default=5, help="加压窗口（秒）")
    parser.add_argument("--think-time", type=float, default=0.05, help="思考时间（秒）")
    parser.add_argument("--report-dir", default="reports", help="报告输出目录")
    args = parser.parse_args()

    httpd = create_server(port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{httpd.server_port}"
    print(f"被测服务已启动：{base_url}")

    sns = seed_devices(base_url, max(args.users, 5))
    print(f"已注册 {len(sns)} 台设备用于参数化")

    stats = Stats()
    ts_base = int(time.time())
    start = time.time() + 0.3
    stop = start + args.duration

    with ThreadPoolExecutor(max_workers=args.users) as pool:
        futures = []
        for i in range(args.users):
            offset = start + (args.ramp * i / max(args.users - 1, 1))
            futures.append(
                pool.submit(worker, base_url, sns, stats, offset, stop, args.think_time, ts_base, i)
            )
        for f in futures:
            f.result()

    result = summarize(stats, start, time.time())
    samples = result.pop("_samples")
    result.pop("_start")

    os.makedirs(args.report_dir, exist_ok=True)
    plot_series(samples, os.path.join(args.report_dir, "perf_throughput.png"))
    report_path = write_report(result, args, args.report_dir)

    os.makedirs(os.path.join(args.report_dir, "raw"), exist_ok=True)
    with open(os.path.join(args.report_dir, "raw", "perf_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: v for k, v in result.items() if k != "scenarios"}, ensure_ascii=False, indent=2))
    print(f"报告已生成：{report_path}")

    httpd.shutdown()
    httpd.server_close()


if __name__ == "__main__":
    main()
