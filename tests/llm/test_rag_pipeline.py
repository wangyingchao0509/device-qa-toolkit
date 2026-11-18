"""RAG 诊断模块的测试。

关注点不是"RAG 答得好不好"，而是：
① 检索在明确的问题上能命中、在范围外的问题上不乱命中；
② 定责规则本身正确（这是整个模块的价值所在）；
③ 四类定责在样例数据里都能被触发出来。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_evals.rag.diagnose import attribute, run_all, ragas_available
from llm_evals.rag.retriever import Retriever

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "llm_evals" / "rag" / "cases.yaml"


@pytest.mark.llm
def test_retriever_hits_expected_doc():
    r = Retriever(top_k=2)
    hits = r.search("退款多久到账")
    assert hits, "明确的问题必须能检索到内容"
    assert hits[0].doc.id == "D2", f"应命中退款文档，实际 {hits[0].doc.id}"


@pytest.mark.llm
def test_retriever_returns_empty_for_out_of_scope():
    """范围外的问题（如"洗澡"）不该硬凑一个文档出来。"""
    r = Retriever(top_k=2)
    assert r.search("洗澡的时候能戴吗") == []


@pytest.mark.llm
@pytest.mark.parametrize(
    "metrics, expected",
    [
        ((0.2, 1.0, 1.0, 1.0), "检索未召回"),
        ((1.0, 0.4, 1.0, 1.0), "检索噪声"),
        ((1.0, 1.0, 0.5, 1.0), "生成幻觉"),
        ((1.0, 1.0, 1.0, 0.3), "答案缺失要点"),
        ((1.0, 1.0, 1.0, 1.0), "通过"),
    ],
    ids=["recall", "precision", "faithfulness", "correctness", "pass"],
)
def test_attribution_rules(metrics, expected):
    """定责顺序必须是：召回 → 精确 → 忠实 → 要点覆盖。"""
    assert attribute(*metrics)[0] == expected


@pytest.mark.llm
def test_all_four_categories_are_represented():
    """样例集必须能触发四类定责，否则等于没演示。"""
    results = run_all(str(CASES), top_k=2)
    got = {r.attribution for r in results}
    for expected in {"检索未召回", "生成幻觉", "答案缺失要点", "通过"}:
        assert expected in got, f"样例集未触发「{expected}」，实际只有 {got}"


@pytest.mark.llm
def test_metrics_in_range():
    for r in run_all(str(CASES), top_k=2):
        for v in (r.context_recall, r.context_precision, r.faithfulness, r.answer_correctness):
            assert 0.0 <= v <= 1.0


@pytest.mark.llm
def test_ragas_skip_reason_is_explicit():
    reason = ragas_available()
    if reason:
        assert "ragas" in reason or "API Key" in reason
