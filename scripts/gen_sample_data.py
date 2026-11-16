"""生成模拟的可靠性试验数据（电池循环 + 温升）。

为什么用生成而不是直接用真实数据：
1. 真实数据涉及产品信息，不便公开；
2. 生成数据可以精确控制"哪台该通过、哪台该失败"，让报告有对照；
3. 故意掺入脏数据，用来演示清洗环节。

运行：python scripts/gen_sample_data.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "sample" / "battery_cycles.csv"
CYCLES = 500
BASE_TS = 1_760_000_000

# sn: (初始容量, 500 次循环后的衰减比例, 温升基准)
PROFILE = {
    "AB10000001": (500, 0.14, 37.0),   # 正常
    "AB10000002": (500, 0.16, 38.0),   # 正常
    "AB10000003": (500, 0.26, 38.5),   # 容量不达标（衰减过快）
    "AB10000004": (505, 0.13, 46.5),   # 温升不达标
    "AB10000005": (498, 0.15, 39.0),   # 正常
}


def gen() -> None:
    rng = random.Random(20261116)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for sn, (initial, fade, temp_base) in PROFILE.items():
        for cycle in range(1, CYCLES + 1):
            ratio = 1 - fade * (cycle / CYCLES)
            capacity = initial * ratio + rng.gauss(0, 1.5)
            temp = temp_base + 3.0 * (cycle / CYCLES) + rng.gauss(0, 0.8)
            rows.append(
                {
                    "sn": sn,
                    "cycle": cycle,
                    "capacity_mah": round(capacity, 2),
                    "temp_c": round(temp, 2),
                    "ts": BASE_TS + cycle * 3600,
                }
            )

    # 故意掺入脏数据，验证清洗逻辑
    rows[10]["capacity_mah"] = ""          # 关键字段缺失
    rows[1200]["temp_c"] = 999             # 超出传感器量程
    rows[1800]["temp_c"] = ""              # 关键字段缺失

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sn", "cycle", "capacity_mah", "temp_c", "ts"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"已生成 {len(rows)} 条记录 -> {OUT}")


if __name__ == "__main__":
    gen()
