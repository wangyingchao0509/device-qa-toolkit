"""可选集成：把黄金集接到 DeepEval 上跑真指标。

设计原则：**可选，且能优雅降级**。
- 没装 deepeval → 返回 skipped，不影响 pytest 全绿；
- 没配 API Key → 同样跳过（真指标必须调判官模型，会产生费用）；
- 装了也配了 → 用 GEval 跑业务判据，输出逐条分数与理由。

单独跑：
    pip install deepeval
    export OPENAI_API_KEY=sk-xxx
    python -m llm_evals.deepeval_adapter --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from llm_evals.provider import get_provider
from llm_evals.runner import load_cases

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "llm_evals" / "cases" / "p0_core.yaml"


def is_installed() -> bool:
    try:
        import deepeval  # noqa: F401
    except Exception:
        return False
    return True


def has_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"))


def skip_reason() -> str:
    if not is_installed():
        return "未安装 deepeval（pip install deepeval）"
    if not has_api_key():
        return "未配置 OPENAI_API_KEY / LLM_API_KEY（真指标需调用判官模型）"
    return ""


def build_criteria(case: dict) -> str:
    """把规则断言翻译成自然语言判据，交给 GEval。"""
    parts: list[str] = []
    if case.get("expected_contains"):
        parts.append("回答必须包含以下要点：" + "、".join(case["expected_contains"]))
    if case.get("must_not_contain"):
        parts.append("回答不得出现以下内容：" + "、".join(case["must_not_contain"]))
    parts.append("不得编造未提供的信息")
    return "；".join(parts) + "。"


def _judge_model():
    """构造判官模型。

    默认走 OpenAI；设置 DEEPEVAL_JUDGE_BASE_URL 后可指向任何 OpenAI 兼容端点：
        export OPENAI_API_KEY=sk-xxx
        export DEEPEVAL_JUDGE_BASE_URL=https://api.deepseek.com/v1
        export DEEPEVAL_JUDGE_MODEL=deepseek-chat
    """
    base_url = os.getenv("DEEPEVAL_JUDGE_BASE_URL")
    if not base_url:
        return None  # 不指定就用 DeepEval 默认（OpenAI）
    from deepeval.models import OpenAIModel

    return OpenAIModel(
        model=os.getenv("DEEPEVAL_JUDGE_MODEL", "deepseek-chat"),
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"),
    )


def _params_enum():
    """不同版本枚举名不同，做兼容。"""
    try:
        from deepeval.test_case import SingleTurnParams

        return SingleTurnParams
    except ImportError:
        from deepeval.test_case import LLMTestCaseParams

        return LLMTestCaseParams


def evaluate_cases(
    cases_path: str | Path = DEFAULT_CASES,
    limit: int | None = None,
    threshold: float = 0.7,
    judge_model: str | None = None,
) -> dict:
    """逐条调用 DeepEval 指标评分，返回汇总。"""
    reason = skip_reason()
    if reason:
        raise RuntimeError(f"DeepEval 不可用：{reason}")

    from deepeval.metrics import AnswerRelevancyMetric, GEval
    from deepeval.test_case import LLMTestCase

    Params = _params_enum()
    judge = _judge_model()
    provider = get_provider()
    cases = load_cases(cases_path)["cases"]
    if limit:
        cases = cases[:limit]

    rows = []
    for case in cases:
        completion = provider.complete(case["input"])
        tc = LLMTestCase(input=case["input"], actual_output=completion.text)

        kwargs = {"threshold": threshold}
        if judge:
            kwargs["model"] = judge
        elif judge_model:
            kwargs["model"] = judge_model

        correctness = GEval(
            name="业务正确性",
            criteria=build_criteria(case),
            evaluation_params=[Params.INPUT, Params.ACTUAL_OUTPUT],
            **kwargs,
        )
        relevancy = AnswerRelevancyMetric(**kwargs)

        correctness.measure(tc)
        relevancy.measure(tc)

        rows.append(
            {
                "id": case["id"],
                "intent": case["intent"],
                "correctness": round(float(correctness.score), 3),
                "relevancy": round(float(relevancy.score), 3),
                "passed": bool(correctness.score >= threshold and relevancy.score >= threshold),
                "reason": getattr(correctness, "reason", "")[:120],
                "answer": completion.text[:80],
            }
        )

    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    return {
        "judge": "deepeval",
        "threshold": threshold,
        "cases": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "avg_correctness": round(sum(r["correctness"] for r in rows) / total, 3) if total else 0.0,
        "avg_relevancy": round(sum(r["relevancy"] for r in rows) / total, 3) if total else 0.0,
        "details": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="用 DeepEval 跑黄金集（可选）")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--limit", type=int, default=5, help="先跑几条试水，控制成本")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--judge-model", default=None, help="判官模型，建议用便宜档")
    args = parser.parse_args()

    reason = skip_reason()
    if reason:
        print(f"跳过：{reason}")
        return

    result = evaluate_cases(args.cases, args.limit, args.threshold, args.judge_model)
    out = ROOT / "reports" / "raw" / "deepeval_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in result.items() if k != "details"}, ensure_ascii=False, indent=2))
    print(f"明细：{out}")


if __name__ == "__main__":
    main()
