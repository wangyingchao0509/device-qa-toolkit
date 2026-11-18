#!/usr/bin/env python3
"""RAG 诊断：跑一遍用例，输出定责矩阵报告。

    python llm_evals/run_rag_eval.py                # 离线指标，零成本
    python llm_evals/run_rag_eval.py --ragas        # 追加 Ragas 真指标（需 Key）
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_evals.rag.diagnose import ragas_available, run_all, run_ragas  # noqa: E402

CASES = ROOT / "llm_evals" / "rag" / "cases.yaml"


def render(results, ragas: dict | None, out_dir: str) -> str:
    total = len(results)
    acts = Counter(r.attribution for r in results)
    passed = acts.get("通过", 0)

    rows = []
    for r in results:
        rows.append(
            f"| {r.id} | {r.question[:12]} | {r.context_recall:.2f} | {r.context_precision:.2f} | "
            f"{r.faithfulness:.2f} | {r.answer_correctness:.2f} | {r.attribution} |"
        )

    issue_rows = []
    for r in results:
        if r.attribution == "通过":
            continue
        issue_rows.append(
            f"| {r.id} | {r.attribution} | {r.advice} | {r.answer[:40]} |"
        )

    lines = [
        "# RAG 诊断报告",
        "",
        "## 一、结论",
        "",
        f"- 用例 **{total}** 条，通过 **{passed}** 条（{passed / total * 100:.0f}%）",
        "- 定责分布：" + "、".join(f"{k} {v} 条" for k, v in acts.most_common()),
        "",
        "| 定责 | 说明 | 该改哪里 |",
        "|---|---|---|",
        "| 检索未召回 | 该找的资料没检索到 | 切分策略 / embedding / query 改写 |",
        "| 检索噪声 | 无关片段挤占上下文 | 加 rerank、加过滤 |",
        "| 生成幻觉 | 资料对但模型自己编 | 加引用约束、要求逐句给依据 |",
        "| 答案缺失要点 | 答了但没答到关键信息 | 拼接策略 / 要点抽取 |",
        "",
        "## 二、逐条指标",
        "",
        "| 用例 | 问题 | Context Recall | Context Precision | Faithfulness | 要点覆盖 | 定责 |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
        "## 三、需要处理的用例",
        "",
    ]
    if issue_rows:
        lines += ["| 用例 | 定责 | 建议 | 实际回答 |", "|---|---|---|---|", *issue_rows]
    else:
        lines.append("全部通过。")

    if ragas:
        lines += [
            "",
            "## 四、Ragas 真指标（判官评分）",
            "",
            "| 指标 | 分数 |",
            "|---|---|",
            *[f"| {k} | {v} |" for k, v in ragas.items()],
        ]

    lines += [
        "",
        "## 五、说明",
        "",
        "- 离线指标用字面重合度近似，**可解释、零成本、结果确定**，适合做回归对比；",
        "- 它不能替代语义判分：同义表述会被判成未覆盖，这是刻意的取舍（宁可误报也不放过）；",
        "- 升级路径：装 `ragas` 并配 Key，加 `--ragas` 得到语义指标，两套结果对照看。",
    ]

    path = os.path.join(out_dir, "rag_report.md")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 诊断")
    parser.add_argument("--cases", default=str(CASES))
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--out", default=str(ROOT / "reports"))
    parser.add_argument("--ragas", action="store_true", help="追加 Ragas 真指标（需 API Key）")
    args = parser.parse_args()

    results = run_all(args.cases, top_k=args.top_k)

    ragas = None
    if args.ragas:
        reason = ragas_available()
        if reason:
            print(f"跳过 Ragas：{reason}")
        else:
            ragas = run_ragas(args.cases)

    path = render(results, ragas, args.out)

    acts = Counter(r.attribution for r in results)
    print(f"用例 {len(results)} 条，通过 {acts.get('通过', 0)} 条")
    for k, v in acts.most_common():
        print(f"  {k}: {v}")
    print(f"报告：{path}")


if __name__ == "__main__":
    main()
