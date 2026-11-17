"""DeepEval 可选集成测试。

设计意图：
- 装了 deepeval 且配了 Key → 真的跑一遍指标；
- 没装 or 没 Key → **skip**，保证别人克隆仓库时 `pytest` 依然全绿。

这是"可选依赖"的标准做法：能力可扩展，但不能让主流程跑不起来。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_evals.deepeval_adapter import (
    build_criteria,
    evaluate_cases,
    has_api_key,
    is_installed,
    skip_reason,
)

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "llm_evals" / "cases" / "p0_core.yaml"


@pytest.mark.llm
def test_criteria_translation_contains_constraints():
    """判据翻译：规则断言必须完整转成自然语言，不能丢约束。"""
    case = {
        "input": "保修期是 30 天对吧",
        "expected_contains": ["12 个月"],
        "must_not_contain": ["30 天", "是的"],
    }
    criteria = build_criteria(case)
    assert "12 个月" in criteria
    assert "30 天" in criteria
    assert "不得编造" in criteria


@pytest.mark.llm
def test_skip_reason_is_explicit():
    """降级原因必须写清楚，否则别人不知道为什么跳过。"""
    reason = skip_reason()
    if is_installed() and has_api_key():
        assert reason == ""
    else:
        assert reason, "不可用时必须给出具体原因"


@pytest.mark.llm
@pytest.mark.skipif(bool(skip_reason()), reason=skip_reason() or "deepeval 不可用")
def test_deepeval_scores_three_cases():
    """真跑：只取 3 条控制成本，验证整条链路通。"""
    result = evaluate_cases(CASES, limit=3, threshold=0.5)
    assert result["cases"] == 3
    assert 0 <= result["avg_correctness"] <= 1
    assert 0 <= result["avg_relevancy"] <= 1
    assert all("reason" in r for r in result["details"]), "每条都要有判官给出的理由"
