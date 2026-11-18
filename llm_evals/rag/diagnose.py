"""RAG 诊断：算四个指标，并给出"该改哪一层"的定责结论。

指标分两套：
  ① 离线可解释指标（默认）：用 key_points 与字面重合度近似，零成本、结果确定；
  ② Ragas 真指标（可选）：装了 ragas 且配了 Key 才算，语义更准但要花钱、有噪声。

两套都只为回答同一个问题：**这次答错，该找检索还是找生成？**
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml

from llm_evals.rag.generator import generate
from llm_evals.rag.retriever import Retriever, bigrams

COVER_THRESHOLD = 0.6   # 要点被判定为"已覆盖"所需的重合度
FAITH_THRESHOLD = 0.8   # 忠实度：几乎每句话都要有依据，标准更严


@dataclass
class RagResult:
    id: str
    question: str
    answer: str
    contexts: list[str]
    context_recall: float
    context_precision: float
    faithfulness: float
    answer_correctness: float
    attribution: str
    advice: str
    missing_points: list[str] = field(default_factory=list)


def coverage(target: str, source_text: str) -> float:
    """target 的字面信息有多少能在 source_text 里找到。"""
    tg = bigrams(target)
    if not tg:
        return 1.0
    sg = bigrams(source_text)
    return len(tg & sg) / len(tg)


def sentence_supported(sent: str, contexts_text: str) -> bool:
    return coverage(sent, contexts_text) >= COVER_THRESHOLD


def point_covered(point: str, contexts_text: str) -> bool:
    return coverage(point, contexts_text) >= COVER_THRESHOLD


def attribute(recall: float, precision: float, faithfulness: float, correctness: float) -> tuple[str, str]:
    """定责规则——这张映射表是整个模块的核心产出。

    判断顺序很重要：先看检索能不能拿到料，再看料排第几，
    最后才看生成是否忠实、是否答到点子上。
    """
    if recall < COVER_THRESHOLD:
        return "检索未召回", "该找的资料没检索到：改切分策略、embedding 模型或做 query 改写"
    if precision < 0.5:
        return "检索噪声", "正确片段没排在首位：加 rerank 或调相似度阈值"
    if faithfulness < FAITH_THRESHOLD:
        return "生成幻觉", "资料找到了但答案夹带无依据内容：加引用约束、要求逐句给依据"
    if correctness < COVER_THRESHOLD:
        return "答案缺失要点", "答了但没答到关键信息：检查拼接策略或让模型抽取要点"
    return "通过", "各环节正常"


def run_case(case: dict, retriever: Retriever) -> RagResult:
    hits = retriever.search(case["question"])
    contexts = [h.doc.text for h in hits]
    ctx_text = "".join(contexts)
    answer = generate(case["question"], contexts)

    points = case.get("key_points", [])
    covered = [p for p in points if point_covered(p, ctx_text)]
    recall = len(covered) / len(points) if points else 1.0

    # 精确率：正确片段排在第一位得 1.0，排第二得 0.5，没检索到得 0
    # （若简单用"有用片段数/检索片段数"，top_k=2 时永远是 0.5，看不出好坏）
    expected = case.get("expected_doc")
    precision = 0.0
    for rank, h in enumerate(hits, start=1):
        if h.doc.id == expected:
            precision = 1.0 / rank
            break

    # 忠实度：答案中被上下文支持的句子占比（"根据资料："前缀不计）
    body = answer.removeprefix("根据资料：")
    sents = [s for s in body.replace("。", "\n").split("\n") if s.strip()]
    supported = [s for s in sents if sentence_supported(s, ctx_text)]
    faithfulness = len(supported) / len(sents) if sents else 0.0

    # 正确性：标准答案的要点是否出现在回答里
    correctness = (
        len([p for p in points if point_covered(p, answer)]) / len(points) if points else 1.0
    )

    act, advice = attribute(recall, precision, faithfulness, correctness)
    return RagResult(
        id=case["id"],
        question=case["question"],
        answer=answer,
        contexts=contexts,
        context_recall=round(recall, 2),
        context_precision=round(precision, 2),
        faithfulness=round(faithfulness, 2),
        answer_correctness=round(correctness, 2),
        attribution=act,
        advice=advice,
        missing_points=[p for p in points if p not in covered],
    )


def run_all(cases_path: str, top_k: int = 2) -> list[RagResult]:
    data = yaml.safe_load(open(cases_path, encoding="utf-8"))
    retriever = Retriever(top_k=top_k)
    return [run_case(c, retriever) for c in data["cases"]]


# ---------------- 可选：Ragas 真指标 ----------------
def ragas_available() -> str:
    try:
        import ragas  # noqa: F401
    except Exception:
        return "未安装 ragas（pip install ragas）"
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")):
        return "未配置 API Key（Ragas 指标需调用判官模型）"
    return ""


def run_ragas(cases_path: str, limit: int = 5) -> dict:
    """用 Ragas 算真指标（faithfulness / context_precision / context_recall / answer_relevancy）。"""
    reason = ragas_available()
    if reason:
        raise RuntimeError(f"Ragas 不可用：{reason}")

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    data = yaml.safe_load(open(cases_path, encoding="utf-8"))
    rows = []
    retriever = Retriever(top_k=2)
    for case in data["cases"][:limit]:
        hits = retriever.search(case["question"])
        contexts = [h.doc.text for h in hits]
        rows.append(
            {
                "question": case["question"],
                "answer": generate(case["question"], contexts),
                "contexts": contexts,
                "ground_truth": case["ground_truth"],
            }
        )

    result = evaluate(
        Dataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return {k: (round(float(v), 3) if isinstance(v, (int, float)) else v) for k, v in result.items()}
