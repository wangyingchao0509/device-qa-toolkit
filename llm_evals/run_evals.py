#!/usr/bin/env python3
"""跑一轮大模型评测并生成报告。

    python llm_evals/run_evals.py                    # 内置 Mock 模型，零依赖可跑
    LLM_API_KEY=xxx python llm_evals/run_evals.py --real
    LLM_BASE_URL=https://api.deepseek.com/v1 LLM_MODEL=deepseek-chat LLM_API_KEY=xxx \
        python llm_evals/run_evals.py --real
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_evals.runner import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="大模型评测")
    parser.add_argument("--cases", default=str(ROOT / "llm_evals" / "cases" / "p0_core.yaml"))
    parser.add_argument("--labels", default=str(ROOT / "llm_evals" / "cases" / "judge_labels.yaml"))
    parser.add_argument("--out", default=str(ROOT / "reports"))
    parser.add_argument("--repeats", type=int, default=3, help="每题重复次数，用于观察稳定性")
    parser.add_argument("--real", action="store_true", help="调用真实模型（需 LLM_API_KEY）")
    args = parser.parse_args()

    result = run(args.cases, args.labels, out_dir=args.out, repeats=args.repeats, use_real=args.real)
    s = result["summary"]
    c = result["calibration"]

    print(f"被测模型：{result['provider']}")
    print(f"通过率：{s['pass_rate'] * 100:.0f}%（{s['passed']}/{s['total_cases']}）")
    print(f"最差分类：{s['worst_intent']}")
    print(f"幻觉 {s['hallucination_count']} 例 / 注入泄露 {s['injection_leak_count']} 例")
    print(f"一致性：{s['consistency'] * 100:.0f}%　平均延迟 {s['avg_latency_ms']:.0f} ms")
    print(f"判官 Kappa：{c['kappa']}（{'可用' if c['usable'] else '不可用'}）")
    print(f"报告：{result['report']}")


if __name__ == "__main__":
    main()
