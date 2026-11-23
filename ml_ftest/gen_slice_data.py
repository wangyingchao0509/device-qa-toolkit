#!/usr/bin/env python3
"""生成切片测试用的样例数据：智能手表活动识别模型的逐条预测结果。

刻意埋了 5 个"坏切片"，用来验证切片分析能不能找出来：
  1. 老年 + 夜间      → 准确率显著低（模拟样本少 + 夜间信号弱）
  2. 25Hz 采样率      → 走/跑混淆严重
  3. 高噪声信号       → 整体准确率下降
  4. 右手佩戴         → 轻微下降（左右手轴向差异）
  5. 众包数据         → 标注噪声（这些错可能是标注错，不是模型错）

同时保留 subject_id —— 阶段①的受试者泄漏检测要用。

    python ml_ftest/gen_slice_data.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ml" / "slice_sample.csv"

DEVICES = ["WATCH-A", "WATCH-B", "WATCH-C"]
POSITIONS = ["左手", "右手"]
RATES = [25, 50, 100]
AGES = ["<30", "30-50", ">50"]
INTENSITY = ["低", "中", "高"]
SOURCES = ["实验室", "众包"]
TOD = ["白天", "夜间"]
QUALITY = ["正常", "高噪声"]
LABELS = ["静止", "走路", "跑步", "骑行"]

BASE_ACC = {"静止": 0.98, "走路": 0.95, "跑步": 0.90, "骑行": 0.86}
CONFUSE = {"走路": "跑步", "跑步": "走路", "骑行": "走路", "静止": "走路"}


def build_rows(n: int = 800, seed: int = 20261123) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        subject = f"S{rng.randint(1, 60):03d}"
        label = rng.choices(LABELS, weights=[0.25, 0.35, 0.25, 0.15])[0]
        device = rng.choices(DEVICES, weights=[0.4, 0.35, 0.25])[0]
        position = rng.choices(POSITIONS, weights=[0.65, 0.35])[0]
        rate = rng.choices(RATES, weights=[0.2, 0.6, 0.2])[0]
        age = rng.choices(AGES, weights=[0.4, 0.4, 0.2])[0]
        intensity = rng.choices(INTENSITY, weights=[0.35, 0.4, 0.25])[0]
        source = rng.choices(SOURCES, weights=[0.6, 0.4])[0]
        tod = rng.choices(TOD, weights=[0.7, 0.3])[0]
        quality = rng.choices(QUALITY, weights=[0.8, 0.2])[0]

        p = BASE_ACC[label]

        # ---- 埋点：让特定切片明显变差 ----
        if age == ">50":
            p -= 0.05
        if age == ">50" and tod == "夜间":
            p -= 0.32           # 埋点 1：老年 + 夜间（组合切片，单维看不出来）
        if rate == 25:
            p -= 0.16           # 埋点 2：低采样率，主要造成走/跑混淆
        if quality == "高噪声":
            p -= 0.18           # 埋点 3：高噪声
        if position == "右手":
            p -= 0.04           # 埋点 4：右手（轻微）
        if source == "众包":
            p -= 0.05           # 埋点 5：众包标注噪声

        p = max(0.05, min(0.99, p))

        if rng.random() < p:
            pred = label
        elif rate == 25 and label in ("走路", "跑步") and rng.random() < 0.7:
            pred = CONFUSE[label]      # 低采样率主要在走/跑之间混（强混淆）
        else:
            pred = rng.choice([x for x in LABELS if x != label])

        rows.append(
            {
                "sample_id": f"SM{i:04d}",
                "subject_id": subject,
                "true_label": label,
                "pred_label": pred,
                "device": device,
                "wear_position": position,
                "sample_rate": rate,
                "age_group": age,
                "activity_intensity": intensity,
                "data_source": source,
                "time_of_day": tod,
                "signal_quality": quality,
            }
        )
    return rows


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    acc = sum(r["true_label"] == r["pred_label"] for r in rows) / len(rows)
    subjects = len({r["subject_id"] for r in rows})
    print(f"已生成 {OUT}")
    print(f"样本 {len(rows)} 条，受试者 {subjects} 人，整体准确率 {acc * 100:.1f}%")


if __name__ == "__main__":
    main()
