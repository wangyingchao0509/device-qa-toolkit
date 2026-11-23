#!/usr/bin/env python3
"""切片分析：按关键维度切开，找最差的那几片。

对应测试计划阶段②：**整体指标达标 ≠ 没有坏掉的部分**。

三条设计原则：
1. 每片都报样本量（n），n < 30 标"样本不足"，不据此下强结论；
2. 用 macro-F1 而不是 accuracy —— 类别失衡时 accuracy 会骗人；
3. 输出"最差切片 + 该找谁改"，而不是只给一张表。

    python ml_ftest/slicing.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ml" / "slice_sample.csv"
LABELS = ["静止", "走路", "跑步", "骑行"]
MIN_N = 30  # 少于这个样本量不下强结论


def per_class_f1(df: pd.DataFrame) -> dict[str, float]:
    """逐类 F1；该类无样本时返回 nan（不参与平均）。"""
    out = {}
    for label in LABELS:
        tp = len(df[(df.true_label == label) & (df.pred_label == label)])
        fp = len(df[(df.true_label != label) & (df.pred_label == label)])
        fn = len(df[(df.true_label == label) & (df.pred_label != label)])
        if tp + fp + fn == 0:
            out[label] = float("nan")
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        out[label] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return out


def slice_stats(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    rows = []
    for value, sub in df.groupby(dim):
        f1s = per_class_f1(sub)
        valid = [v for v in f1s.values() if v == v]
        acc = (sub.true_label == sub.pred_label).mean()
        worst_label = min(f1s.items(), key=lambda kv: (kv[1] if kv[1] == kv[1] else 9))[0]
        rows.append(
            {
                "维度": dim,
                "取值": value,
                "n": len(sub),
                "accuracy": round(acc, 3),
                "macro_f1": round(sum(valid) / len(valid), 3) if valid else 0.0,
                "最差类": worst_label,
                "最差类F1": round(f1s[worst_label], 3),
                "样本充足": "是" if len(sub) >= MIN_N else "否",
            }
        )
    return pd.DataFrame(rows)


def confusion_top(df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    wrong = df[df.true_label != df.pred_label]
    pairs = (
        wrong.groupby(["true_label", "pred_label"]).size().reset_index(name="count")
        .sort_values("count", ascending=False).head(k)
    )
    pairs["占比"] = (pairs["count"] / len(wrong) * 100).round(1).astype(str) + "%"
    return pairs.rename(columns={"true_label": "真实", "pred_label": "预测", "count": "次数"})


def render(df: pd.DataFrame, out_dir: Path) -> Path:
    overall_acc = (df.true_label == df.pred_label).mean()
    overall_f1 = per_class_f1(df)
    valid = [v for v in overall_f1.values() if v == v]
    overall_macro = sum(valid) / len(valid)

    # 组合切片：单维看不出来的问题，交叉之后会暴露
    df = df.copy()
    df["age_night"] = df["age_group"] + "-" + df["time_of_day"]

    dims = [
        "device", "wear_position", "sample_rate", "age_group", "age_night",
        "activity_intensity", "data_source", "time_of_day", "signal_quality",
    ]
    all_slices = pd.concat([slice_stats(df, d) for d in dims], ignore_index=True)
    reliable = all_slices[all_slices["样本充足"] == "是"].sort_values("macro_f1")
    worst = reliable.head(5)
    threshold = round(overall_macro - 0.15, 3)

    def verdict(f1: float) -> str:
        if f1 < threshold - 0.05:
            return "**必须修**"
        if f1 < threshold:
            return "需给结论"
        return "可接受"

    lines = [
        "# 切片分析报告",
        "",
        "## 一、结论",
        "",
        f"- 样本 **{len(df)}** 条，受试者 **{df.subject_id.nunique()}** 人",
        f"- 整体准确率 **{overall_acc * 100:.1f}%**，macro-F1 **{overall_macro:.3f}**",
        f"- 共切 **{len(all_slices)}** 片，其中样本充足（n≥{MIN_N}）**{len(reliable)}** 片",
        "",
        "### 最差 5 个切片（只统计 n≥30 的切片）",
        "",
        "| 维度 | 取值 | n | accuracy | macro-F1 | 最差类 | 结论 | 该找谁改 |",
        "|---|---|---|---|---|---|---|---|",
    ]

    owner_hint = {
        "device": "硬件/固件",
        "wear_position": "算法（轴向处理）",
        "sample_rate": "数据/预处理",
        "age_group": "数据（补样本）",
        "activity_intensity": "算法",
        "data_source": "标注",
        "time_of_day": "数据（补样本）",
        "signal_quality": "算法（滤波）",
        "age_night": "数据（补老年夜间样本）",
    }
    for _, r in worst.iterrows():
        lines.append(
            f"| {r['维度']} | {r['取值']} | {r['n']} | {r['accuracy']:.1%} | "
            f"{r['macro_f1']:.3f} | {r['最差类']} | {verdict(r['macro_f1'])} | "
            f"{owner_hint.get(r['维度'], '待定')} |"
        )

    lines += [
        "",
        f"> 判据：任一 P0 切片的 macro-F1 不应低于整体（{overall_macro:.3f}）− 0.15，"
        f"即 **{threshold:.3f}**。低于此线需给出结论：修 / 接受 / 补数据。",
        f"> 实际**低于判据线 {len(reliable[reliable['macro_f1'] < threshold])} 个切片**，"
        f"其中明显越线（低于 {threshold - 0.05:.3f}）"
        f"**{len(reliable[reliable['macro_f1'] < threshold - 0.05])} 个**。",
        "",
        "> 注意：最差的切片往往是**组合切片**（如「年龄 × 时间段」）。",
        "> 单维切片会把问题稀释到看不出来——这也是为什么真实项目里要主动构造交叉维度。",
        "",
        "## 二、混淆热点",
        "",
        "| 真实 | 预测 | 次数 | 占全部错误 |",
        "|---|---|---|---|",
    ]
    for _, r in confusion_top(df).iterrows():
        lines.append(f"| {r['真实']} | {r['预测']} | {r['次数']} | {r['占比']} |")

    lines += [
        "",
        "## 三、逐类指标",
        "",
        "| 类别 | F1 |",
        "|---|---|",
    ]
    for label, f1 in overall_f1.items():
        lines.append(f"| {label} | {'—' if f1 != f1 else f'{f1:.3f}'} |")

    lines += [
        "",
        "## 四、全量切片明细",
        "",
        "| 维度 | 取值 | n | accuracy | macro-F1 | 最差类 | 结论 | 样本充足 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in all_slices.sort_values(["维度", "macro_f1"]).iterrows():
        lines.append(
            f"| {r['维度']} | {r['取值']} | {r['n']} | {r['accuracy']:.1%} | "
            f"{r['macro_f1']:.3f} | {r['最差类']} | {verdict(r['macro_f1'])} | {r['样本充足']} |"
        )

    lines += [
        "",
        "## 五、怎么用这份报告",
        "",
        "1. **先看第一节**：最差切片就是风险清单，每条都要有处置结论；",
        "2. **再看第二节**：混淆热点指向具体问题（走/跑混淆 → 采样率或特征窗口）；",
        "3. **第三节**：某个类 F1 低但整体高，说明它被平均掉了；",
        "4. **第四节**：`样本充足=否` 的行不能当缺陷报，要写\"样本不足，需补数据\"。",
        "",
        "> 关键提醒：**不要只看整体 accuracy。** 本数据整体 {:.1%}，"
        "但最差切片只有 {:.1%}，相差 {:.0f} 个百分点。".format(
            overall_acc, worst.iloc[0]["accuracy"], (overall_acc - worst.iloc[0]["accuracy"]) * 100
        ),
    ]

    path = out_dir / "ml_slice_report.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="切片分析")
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--out", default=str(ROOT / "reports"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    path = render(df, Path(args.out))

    acc = (df.true_label == df.pred_label).mean()
    print(f"样本 {len(df)} 条，整体准确率 {acc * 100:.1f}%")
    print(f"报告：{path}")


if __name__ == "__main__":
    main()
