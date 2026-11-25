"""评测执行器：跑批 → 断言 → 汇总 → 报告。

四个必须记录的东西（少一个，结论下周就无法复现）：
  模型版本、提示词版本、数据集版本、采样参数（temperature / seed）
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from llm_evals.judge import rule_judge
from llm_evals.provider import Completion, get_provider

DATASET_VERSION = "golden-20261117"
PROMPT_VERSION = "prompt-v1.0.0"

# 示例价格（美元 / 百万 token），按实际供应商改
PRICE_INPUT_PER_M = 2.0
PRICE_OUTPUT_PER_M = 10.0


@dataclass
class CaseResult:
    id: str
    intent: str
    severity: str
    input: str
    answer: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    total_tokens: int = 0
    run_index: int = 0


def load_cases(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {"version": data.get("version", "unknown"), "cases": data["cases"]}


def run_case(provider, case: dict, run_index: int = 0) -> CaseResult:
    completion: Completion = provider.complete(case["input"])
    verdict = rule_judge(
        completion.text,
        case.get("expected_contains", []),
        case.get("must_not_contain", []),
    )
    return CaseResult(
        id=case["id"],
        intent=case["intent"],
        severity=case.get("severity", "P1"),
        input=case["input"],
        answer=completion.text,
        passed=verdict.passed,
        reasons=verdict.reasons,
        latency_ms=round(completion.latency_ms, 1),
        total_tokens=completion.total_tokens,
        run_index=run_index,
    )


def run_all(provider, cases: list[dict], repeats: int = 3) -> list[CaseResult]:
    """每题跑 repeats 次，用于观察稳定性（同一问题多次回答是否一致）。"""
    results: list[CaseResult] = []
    for case in cases:
        for i in range(repeats):
            results.append(run_case(provider, case, run_index=i))
    return results


def aggregate(results: list[CaseResult]) -> dict:
    by_case = defaultdict(list)
    for r in results:
        by_case[r.id].append(r)

    # 首轮结果作为主口径；多轮用于稳定性
    first_round = [rs[0] for rs in by_case.values()]
    total = len(first_round)
    passed = sum(1 for r in first_round if r.passed)

    by_intent = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in first_round:
        by_intent[r.intent]["total"] += 1
        by_intent[r.intent]["passed"] += int(r.passed)

    for v in by_intent.values():
        v["pass_rate"] = round(v["passed"] / v["total"], 3) if v["total"] else 0.0

    # 稳定性：同一题多次结果是否一致
    stable = sum(1 for rs in by_case.values() if len({r.passed for r in rs}) == 1)
    consistency = round(stable / len(by_case), 3) if by_case else 0.0

    # 幻觉与注入单列，这两类要单独向主管汇报
    hallucination = [r for r in first_round if r.intent == "事实准确" and not r.passed]
    injection = [r for r in first_round if r.intent == "提示词注入" and not r.passed]

    tokens = sum(r.total_tokens for r in results)
    cost = tokens / 1_000_000 * max(PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M)

    return {
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "worst_intent": min(by_intent.items(), key=lambda kv: kv[1]["pass_rate"])[0] if by_intent else None,
        "by_intent": dict(sorted(by_intent.items(), key=lambda kv: kv[1]["pass_rate"])),
        "consistency": consistency,
        "hallucination_count": len(hallucination),
        "injection_leak_count": len(injection),
        "avg_latency_ms": round(sum(r.latency_ms for r in results) / len(results), 1) if results else 0,
        "total_tokens": tokens,
        "estimated_cost_usd": round(cost, 4),
        "failures": [
            {
                "id": r.id,
                "intent": r.intent,
                "severity": r.severity,
                "input": r.input,
                "answer": r.answer,
                "reasons": r.reasons,
            }
            for r in first_round
            if not r.passed
        ],
    }


def save_run(summary: dict, results: list[CaseResult], provider_name: str, out_dir: str) -> str:
    os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
    payload = {
        "run_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "model": provider_name,
        "prompt_version": PROMPT_VERSION,
        "dataset_version": DATASET_VERSION,
        "temperature": 0.2,
        "seed": 42,
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    path = os.path.join(out_dir, "raw", "llm_eval_run.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def plot_by_intent(summary: dict, out_path: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3})

    label_en = {
        "退换货": "Returns",
        "保修": "Warranty",
        "故障排查": "Troubleshooting",
        "续航充电": "Battery & Charging",
        "数据隐私": "Data Privacy",
        "越界拒答": "Out-of-scope",
        "提示词注入": "Prompt Injection",
        "事实准确": "Factuality",
    }
    items = list(summary["by_intent"].items())
    names = [label_en.get(k, k) for k, _ in items]
    rates = [v["pass_rate"] * 100 for _, v in items]

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    colors = ["#c0392b" if r < 100 else "#1f4e79" for r in rates]
    ax.barh(names, rates, color=colors, height=0.55)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Pass rate (%)")
    ax.set_title("Pass rate by intent")
    for i, r in enumerate(rates):
        ax.text(r + 1.5, i, f"{r:.0f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def render_report(summary: dict, calibration: dict, provider_name: str, fig_path: str, out_dir: str) -> str:
    rows = []
    for name, v in summary["by_intent"].items():
        flag = "PASS" if v["pass_rate"] == 1 else "**FAIL**"
        rows.append(f"| {name} | {v['passed']}/{v['total']} | {v['pass_rate'] * 100:.0f}% | {flag} |")

    fail_rows = []
    for f in summary["failures"]:
        fail_rows.append(
            f"| {f['id']} | {f['severity']} | {f['input'][:18]} | {f['answer'][:40]} | {'；'.join(f['reasons'])} |"
        )

    lines = [
        "# 大模型评测报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 被测模型：`{provider_name}`",
        f"- 数据集版本：`{DATASET_VERSION}`　提示词版本：`{PROMPT_VERSION}`　temperature=0.2　seed=42",
        "",
        "## 一、结论",
        "",
        f"- 用例 {summary['total_cases']} 条，通过 **{summary['passed']}**，"
        f"未通过 **{summary['failed']}**，通过率 **{summary['pass_rate'] * 100:.0f}%**",
        f"- 最差分类：**{summary['worst_intent']}**",
        f"- 幻觉 **{summary['hallucination_count']}** 例，提示词注入泄露 **{summary['injection_leak_count']}** 例",
        f"- 多轮一致性：**{summary['consistency'] * 100:.0f}%**（同题多次结果是否稳定）",
        f"- 平均延迟 **{summary['avg_latency_ms']:.0f} ms**，总 token **{summary['total_tokens']}**，"
        f"估算成本 **${summary['estimated_cost_usd']}**",
        "",
        "## 二、分类通过率",
        "",
        "| 分类 | 通过/总数 | 通过率 | 结论 |",
        "|---|---|---|---|",
        *rows,
        "",
        f"![pass rate]({os.path.basename(fig_path)})",
        "",
        "## 三、未通过明细",
        "",
    ]

    if fail_rows:
        lines += [
            "| 用例 | 级别 | 输入 | 实际回答 | 失败原因 |",
            "|---|---|---|---|---|",
            *fail_rows,
        ]
    else:
        lines.append("本轮无未通过用例。")

    lines += [
        "",
        "## 四、判官校准",
        "",
        f"- 校准样本 **{calibration['n']}** 条，人工与判官观察一致率 **{calibration['agreement'] * 100:.0f}%**",
        f"- **Cohen's Kappa = {calibration['kappa']}**，"
        f"bootstrap 95% CI **[{calibration['ci_low']:.2f}, {calibration['ci_high']:.2f}]**（阈值 0.6）",
        f"- 结论：**{calibration['conclusion']}**",
        f"  - bootstrap 重采样里 **{calibration['below_ratio'] * 100:.0f}%** 低于阈值；"
        + (
            f"要让区间下限站上 0.6，样本约需扩到 **{calibration['recommend_n']}** 条"
            if calibration.get("recommend_n") and calibration["recommend_n"] > calibration["n"]
            else "样本量已足以让下限高于阈值"
        ),
        "",
        "> 为什么不能只报点估计：Kappa 的分母含 Pe，没有闭式方差，"
        "小样本直接套比例的公式会算出超过 1 的区间。"
        f"{calibration['n']} 条样本的 0.756 与 200 条的 0.756 不是一回事。",
    ]
    if calibration["disagreements"]:
        lines += ["", "分歧样本（判官判错的地方）："]
        for d in calibration["disagreements"]:
            lines.append(
                f"- `{d['id']}` 人工={d['human']} 判官={d['judge']} 分数={d['score']}：{d['answer']}"
            )

    lines += [
        "",
        "## 五、复现方式",
        "",
        "```bash",
        "python llm_evals/run_evals.py            # 默认使用内置 Mock 模型",
        "LLM_API_KEY=xxx python llm_evals/run_evals.py --real   # 换真实模型",
        "```",
        "",
        "> 记录里已绑定 模型 / 提示词 / 数据集 三个版本，换任何一个都必须重新跑基线，",
        "> 否则分数不可比。",
    ]

    path = os.path.join(out_dir, "llm_eval_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def run(cases_path: str, labels_path: str, out_dir: str = "reports", repeats: int = 3, use_real: bool = False) -> dict:
    from llm_evals.judge import calibrate

    provider = get_provider(use_real=use_real)
    data = load_cases(cases_path)
    results = run_all(provider, data["cases"], repeats=repeats)
    summary = aggregate(results)

    os.makedirs(out_dir, exist_ok=True)
    fig = plot_by_intent(summary, os.path.join(out_dir, "llm_pass_rate.png"))
    calibration = calibrate(labels_path)

    save_run(summary, results, provider.name, out_dir)
    report = render_report(summary, calibration, provider.name, fig, out_dir)

    return {"summary": summary, "calibration": calibration, "report": report, "provider": provider.name}
