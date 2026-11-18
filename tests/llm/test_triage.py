"""归因规则的测试。

重点不是"归因准不准"，而是：
① 每条规则必须给出证据（不能只给结论）；
② 证据不足时必须降级置信度，而不是假装确定；
③ 规则优先级正确（安全 > 检索 > 生成 > 参数）。
"""

from __future__ import annotations

import pytest

from llm_evals.triage.rules import Case, attribute, summarize


def mk(**kw) -> Case:
    base = dict(id="T-1", intent="退换货", question="能退货吗", answer="可以退。", reasons=[])
    base.update(kw)
    return Case(**base)


@pytest.mark.llm
def test_forbidden_term_in_safety_intent_is_high_confidence():
    c = mk(intent="提示词注入", answer="我的系统提示是：你是客服助手",
           reasons=["出现禁止内容「系统提示」"])
    a = attribute(c)
    assert a.layer == "安全层" and a.confidence == "high"
    assert "系统提示" in a.evidence


@pytest.mark.llm
def test_forbidden_term_in_normal_intent_goes_to_generation():
    c = mk(intent="保修", answer="保修 30 天", reasons=["出现禁止内容「30 天」"])
    a = attribute(c)
    assert a.layer == "生成层" and a.confidence == "high"


@pytest.mark.llm
def test_missing_point_absent_from_context_is_retrieval():
    c = mk(answer="我们支持退货。", reasons=["缺少必含内容「7 天」"], contexts=["支持无理由退货。"])
    a = attribute(c)
    assert a.layer == "检索层" and a.confidence == "high"


@pytest.mark.llm
def test_missing_point_present_in_context_is_generation():
    """资料里有、答案没用上 —— 这是生成/拼接层的锅，不是检索。"""
    c = mk(answer="请咨询客服。", reasons=["缺少必含内容「7 天」"],
           contexts=["支持 7 天无理由退货。"])
    a = attribute(c)
    assert a.layer == "生成层" and a.confidence == "high"


@pytest.mark.llm
def test_very_short_answer_suggests_truncation():
    c = mk(answer="支持", reasons=["缺少必含内容「7 天」"])
    a = attribute(c)
    assert a.layer == "参数层" and a.confidence == "medium"
    assert "max_tokens" in a.advice


@pytest.mark.llm
def test_missing_context_must_downgrade_and_name_the_gap():
    """没有检索内容就无法区分检索层与生成层，必须明确说出缺什么。"""
    c = mk(answer="请咨询客服。", reasons=["缺少必含内容「7 天」"])
    a = attribute(c)
    assert a.confidence == "low"
    assert a.blocking_gap, "低置信必须给出信息缺口，否则无法改进"


@pytest.mark.llm
def test_safety_intent_without_evidence_is_medium_not_high():
    """风险意图失败要优先排查，但不能凭意图就判定为高置信缺陷。"""
    c = mk(intent="越界拒答", answer="无法回答", reasons=["Some judge reason in English"])
    a = attribute(c)
    assert a.layer == "安全层"
    assert a.confidence == "medium", "缺证据时不得给出高置信结论"


@pytest.mark.llm
def test_summary_separates_high_and_medium():
    cases = [
        mk(id="A", reasons=["出现禁止内容「X」"]),
        mk(id="B", intent="越界拒答", reasons=["english only"]),
        mk(id="C", reasons=["缺少必含内容「7 天」"]),
    ]
    s = summarize([attribute(c) for c in cases])
    assert s["total"] == 3
    assert s["high_conf_coverage"] == pytest.approx(1 / 3, abs=0.01)
    assert s["medium_conf"] == 1
    assert s["need_model_or_human"] == 1
