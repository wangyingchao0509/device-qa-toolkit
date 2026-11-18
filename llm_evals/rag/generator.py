"""RAG 生成器：从检索到的上下文拼答案，并**故意注入两类典型缺陷**。

为什么要故意注入：
真实的 RAG 故障不是随机的，就那几类——没检索到、检索到噪声、检索对了却自己编。
注入之后，定责矩阵才有东西可诊断，读者也能验证"工具确实抓得住"。

FLAW 只对指定问题生效，其余问题都是忠实拼接，便于对比。
"""

from __future__ import annotations

import re

from llm_evals.rag.retriever import bigrams

SENT_SPLIT = re.compile(r"[。；！？]")

# 故意注入的缺陷：question -> 追加的无依据断言
FLAW: dict[str, str] = {
    # 幻觉：资料里没有"延长保修"，模型自己编了
    "保修期能延长吗": "另外，购买延保服务后保修期可延长至 30 个月。",
    # 事实错误：把 12 个月说成 24 个月
    "屏幕碎了能保修吗": "屏幕属于保修范围，可免费更换。",
}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def generate(question: str, contexts: list[str]) -> str:
    """朴素 RAG：按与问题的字面重合度挑句子拼接。"""
    q = bigrams(question)
    scored: list[tuple[float, str]] = []
    for ctx in contexts:
        for sent in split_sentences(ctx):
            sg = bigrams(sent)
            if not sg:
                continue
            overlap = len(q & sg) / (len(q) or 1)
            scored.append((overlap, sent))

    scored.sort(key=lambda x: -x[0])
    picked = [s for ov, s in scored[:2] if ov > 0]
    answer = "根据资料：" + "。".join(picked) + "。" if picked else "资料中没有找到相关信息。"

    extra = FLAW.get(question)
    if extra:
        answer += extra
    return answer
