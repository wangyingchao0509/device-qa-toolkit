#!/usr/bin/env python3
"""Bad Case 归因工具：读评测结果 → 输出归因表 + 覆盖率。

    python -m llm_evals.triage.analyze                    # 自动找 reports/raw 下的结果
    python -m llm_evals.triage.analyze --file xxx.json

支持两种输入（自动识别）：
  · llm_eval_run.json    自带判据（expected_contains / must_not_contain）→ 归因精确
  · deepeval_result.json 只有分数与理由 → 多数条目会落到"证据不足"

这个差异本身就是产品的第一个洞察：**归因质量取决于评测结果带了多少上下文**。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from llm_evals.triage.rules import Case, attribute, summarize  # noqa: E402

RAW = ROOT / "reports" / "raw"


def _dedupe(cases: list[Case]) -> list[Case]:
    """同一用例跑了多轮时只归因一次，否则统计会被重复项放大。"""
    seen: set[str] = set()
    out: list[Case] = []
    for c in cases:
        if c.id in seen:
            continue
        seen.add(c.id)
        out.append(c)
    return out


def load_cases(path: Path) -> tuple[list[Case], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases: list[Case] = []

    if "results" in data:  # llm_eval_run.json：带 reasons
        for r in data["results"]:
            if r.get("passed"):
                continue
            cases.append(
                Case(
                    id=r["id"], intent=r.get("intent", ""), question=r.get("input", ""),
                    answer=r.get("answer", ""), reasons=r.get("reasons", []),
                    severity=r.get("severity", "P1"),
                )
            )
        return _dedupe(cases), "llm_eval_run（含判据原文）"

    if "details" in data:  # deepeval_result.json：只有分数与理由
        for d in data["details"]:
            if d.get("passed"):
                continue
            cases.append(
                Case(
                    id=d["id"], intent=d.get("intent", ""), question=d.get("input", ""),
                    answer=d.get("answer", ""), reasons=[d.get("reason", "")],
                    severity="P1",
                )
            )
        return _dedupe(cases), "deepeval_result（仅分数与理由）"

    return cases, "未知格式"


def render(cases: list[Case], source: str, summary: dict, out_dir: Path) -> Path:
    rows = []
    for c in cases:
        a = attribute(c)
        rows.append(
            f"| {a.case_id} | {c.intent} | **{a.layer}** | {a.verdict} | {a.evidence[:40]} | "
            f"{a.owner} | {a.confidence} |"
        )

    gap_rows = [
        f"| {a.case_id} | {a.blocking_gap} |"
        for a in (attribute(c) for c in cases)
        if a.blocking_gap
    ]

    lines = [
        "# Bad Case 归因报告",
        "",
        f"- 数据来源：`{source}`",
        f"- 失败用例：**{summary['total']}** 条",
        f"- 规则可判定：**{summary['rule_coverage'] * 100:.0f}%**"
        f"（{summary['total'] - summary['need_model_or_human']}/{summary['total']}）",
        f"  - 其中高置信 **{summary['high_conf_coverage'] * 100:.0f}%**、"
        f"中置信 {summary['medium_conf']} 条（可定位层，但需人工确认失效点）",
        f"- 需人工兜底：**{summary['need_model_or_human']}** 条",
        "",
        "## 一、归因分布",
        "",
        "| 层 | 条数 |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in summary["by_layer"].items()],
        "",
        "## 二、逐条归因",
        "",
        "| 用例 | 意图 | 归因层 | 结论 | 证据 | 建议谁改 | 置信度 |",
        "|---|---|---|---|---|---|---|",
        *rows,
    ]

    if gap_rows:
        lines += [
            "",
            "## 三、信息缺口（为什么判不了）",
            "",
            "| 用例 | 缺什么 |",
            "|---|---|",
            *gap_rows,
            "",
            "> 结论：**归因质量取决于评测结果带了多少上下文**。",
            "> 结果文件里只有分数和理由时，规则只能覆盖一小部分；带上判据原文与检索片段后，覆盖率显著提升。",
        ]

    path = out_dir / "triage_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Bad Case 归因")
    parser.add_argument("--file", default=None, help="评测结果 JSON；默认自动查找")
    parser.add_argument("--out", default=str(ROOT / "reports"))
    args = parser.parse_args()

    target = Path(args.file) if args.file else None
    if target is None:
        for name in ("llm_eval_run.json", "deepeval_result.json"):
            if (RAW / name).exists():
                target = RAW / name
                break
    if target is None or not target.exists():
        print("未找到评测结果，请先跑 make llm-evals")
        raise SystemExit(1)

    cases, source = load_cases(target)
    if not cases:
        print(f"{target.name} 中没有失败用例。")
        return

    summary = summarize([attribute(c) for c in cases])
    path = render(cases, source, summary, Path(args.out))

    print(f"数据源：{source}")
    print(f"失败用例 {summary['total']} 条，规则覆盖 {summary['rule_coverage'] * 100:.0f}%"
          f"（高置信 {summary['high_conf_coverage'] * 100:.0f}%），需兜底 {summary['need_model_or_human']} 条")
    print("归因分布：" + "、".join(f"{k} {v}" for k, v in summary["by_layer"].items()))
    print(f"报告：{path}")


if __name__ == "__main__":
    main()
