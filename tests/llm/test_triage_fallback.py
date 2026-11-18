"""模型兜底层的测试。

四条约束逐条验证 —— 这几条比"能调通模型"重要得多：
① 只处理 low/medium；② 输出必须过校验；③ 不得自称高置信；④ 失败要降级不中断。
"""

from __future__ import annotations

import pytest

from llm_evals.triage.llm_fallback import (
    MockJudge,
    ModelTriage,
    parse_model_output,
)
from llm_evals.triage.rules import Case, attribute


def mk(**kw) -> Case:
    base = dict(id="T-1", intent="保修", question="保修多久", answer="请咨询客服。", reasons=[])
    base.update(kw)
    return Case(**base)


@pytest.mark.llm
def test_parse_handles_fenced_json():
    raw = '```json\n{"layer":"生成层","verdict":"漏要点","evidence":"e","confidence":"medium"}\n```'
    assert parse_model_output(raw)["layer"] == "生成层"


@pytest.mark.llm
def test_parse_rejects_unknown_layer():
    raw = '{"layer":"模型瞎编的层","verdict":"x","confidence":"high"}'
    assert parse_model_output(raw) is None, "层名必须过白名单，否则会污染统计"


@pytest.mark.llm
def test_parse_rejects_garbage():
    assert parse_model_output("我觉得是生成的问题") is None
    assert parse_model_output("") is None


@pytest.mark.llm
def test_model_cannot_claim_high_confidence():
    class Liar:
        def complete(self, prompt: str) -> str:
            return '{"layer":"生成层","verdict":"x","evidence":"e","confidence":"high"}'

    case = mk(reasons=["缺少必含内容「12 个月」"])
    base = attribute(case)
    got = ModelTriage(provider=Liar()).judge(case, base)
    assert got.confidence == "medium", "模型没有证据链，不允许给高置信"


@pytest.mark.llm
def test_high_confidence_is_untouched():
    """已经高置信的结论不能被模型改写，否则规则的价值就没了。"""
    class Spy:
        called = False

        def complete(self, prompt: str) -> str:
            Spy.called = True
            return '{"layer":"参数层","verdict":"x","evidence":"e","confidence":"medium"}'

    case = mk(answer="保修 30 天", reasons=["出现禁止内容「30 天」"])
    base = attribute(case)
    assert base.confidence == "high"
    got = ModelTriage(provider=Spy()).judge(case, base)
    assert got.layer == base.layer and not Spy.called


@pytest.mark.llm
def test_provider_exception_keeps_rule_result():
    class Boom:
        def complete(self, prompt: str) -> str:
            raise TimeoutError("模拟超时")

    case = mk(reasons=["缺少必含内容「12 个月」"])
    base = attribute(case)
    got = ModelTriage(provider=Boom()).judge(case, base)
    assert got.layer == base.layer
    assert "保留规则结论" in got.evidence


@pytest.mark.llm
def test_mock_judge_is_deterministic():
    case = mk(intent="事实准确", reasons=["english only"])
    base = attribute(case)
    a = ModelTriage(provider=MockJudge()).judge(case, base)
    b = ModelTriage(provider=MockJudge()).judge(case, base)
    assert a.layer == b.layer and a.source == "llm"


@pytest.mark.llm
def test_source_is_recorded():
    """必须能区分规则判的还是模型判的，否则无法评估模型值不值。"""
    case = mk(intent="提示词注入", reasons=["english only"])
    got = ModelTriage(provider=MockJudge()).judge(case, attribute(case))
    assert got.source in {"rule", "llm"}
