"""报告生成：Markdown 汇总，含判定结论、明细表与图表。

报告是测试人员的产出物。这里刻意把「结论」放最前面，
因为主管看报告只关心三件事：过没过、谁没过、为什么。
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from analysis.metrics import (
    CAPACITY_PASS_RATIO,
    TEMP_PASS_LIMIT_C,
    overall_summary,
    per_device_metrics,
)


def _table(metrics: pd.DataFrame) -> str:
    header = "| SN | 循环数 | 初始容量 | 末次容量 | 容量保持率 | P95 温升 | 最高温升 | 容量判定 | 温升判定 | 结论 |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    rows = []
    for _, r in metrics.iterrows():
        rows.append(
            "| {sn} | {cycles} | {initial_mah} | {last_mah} | {retention_pct}% | {p95_temp_c}℃ | "
            "{max_temp_c}℃ | {c} | {t} | {o} |".format(
                sn=r["sn"],
                cycles=r["cycles"],
                initial_mah=r["initial_mah"],
                last_mah=r["last_mah"],
                retention_pct=r["retention_pct"],
                p95_temp_c=r["p95_temp_c"],
                max_temp_c=r["max_temp_c"],
                c="PASS" if r["capacity_pass"] else "**FAIL**",
                t="PASS" if r["temp_pass"] else "**FAIL**",
                o="PASS" if r["overall_pass"] else "**FAIL**",
            )
        )
    return "\n".join([header, sep, *rows])


def build_report(
    raw_count: int,
    clean_df: pd.DataFrame,
    rejected: pd.DataFrame,
    metrics: pd.DataFrame,
    figures: dict[str, str],
    out_path: str,
) -> str:
    summary = overall_summary(metrics)
    failed = metrics[~metrics["overall_pass"]]

    lines = [
        "# 可靠性试验数据分析报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 数据来源：`data/sample/battery_cycles.csv`",
        f"- 原始记录 {raw_count} 条，清洗后 {len(clean_df)} 条，剔除 {len(rejected)} 条",
        "",
        "## 一、结论",
        "",
        f"- 受试设备 **{summary['devices']}** 台，通过 **{summary['passed']}** 台，"
        f"未通过 **{summary['failed']}** 台，通过率 **{summary['pass_rate_pct']}%**",
        f"- 最差容量保持率：**{summary['worst_retention_pct']}%**（判据 ≥ {CAPACITY_PASS_RATIO * 100:.0f}%）",
        f"- 最高 P95 温升：**{summary['highest_p95_temp_c']}℃**（判据 ≤ {TEMP_PASS_LIMIT_C}℃）",
    ]

    if not failed.empty:
        lines += ["", "**未通过设备及原因：**"]
        for _, r in failed.iterrows():
            reasons = []
            if not r["capacity_pass"]:
                reasons.append(f"容量保持率 {r['retention_pct']}% 低于 {CAPACITY_PASS_RATIO * 100:.0f}%")
            if not r["temp_pass"]:
                reasons.append(f"P95 温升 {r['p95_temp_c']}℃ 超过 {TEMP_PASS_LIMIT_C}℃")
            lines.append(f"- `{r['sn']}`：{'；'.join(reasons)}")
    else:
        lines += ["", "全部设备通过判定。", ""]

    lines += [
        "",
        "## 二、明细",
        "",
        _table(metrics),
        "",
        "## 三、图表",
        "",
        f"![容量衰减]({os.path.basename(figures['capacity'])})",
        "",
        f"![温升分布]({os.path.basename(figures['temp'])})",
        "",
        f"![判定概览]({os.path.basename(figures['pass'])})",
        "",
        "## 四、数据质量问题",
        "",
    ]

    if rejected.empty:
        lines.append("本次无被剔除记录。")
    else:
        by_reason = rejected.groupby("reason").size().to_dict()
        lines.append(f"共剔除 {len(rejected)} 条，明细：")
        for reason, count in by_reason.items():
            label = "关键字段缺失" if reason == "missing_field" else "超出传感器量程"
            lines.append(f"- {label}：{count} 条")
        lines += ["", "> 清洗规则说明：缺失与超量程记录一律剔除并计数，不做插补——",
                  "> 插补会让「看起来通过」的风险变高，测试报告应如实反映数据缺失。"]

    content = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return out_path


def run(csv_path: str, out_dir: str = "reports") -> str:
    from analysis import metrics as m
    from analysis import plots

    raw = m.load_readings(csv_path)
    clean_df, rejected = m.clean(raw)
    metrics_df = m.per_device_metrics(clean_df)
    os.makedirs(out_dir, exist_ok=True)
    figures = plots.make_all(clean_df, metrics_df, out_dir)
    return build_report(
        len(raw), clean_df, rejected, metrics_df, figures, os.path.join(out_dir, "report.md")
    )
