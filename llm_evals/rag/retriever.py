"""极简检索器：中文字符二元组 + 逆文档频率加权。

为什么不用向量库：
1. 零依赖、结果确定，适合做回归对比（换 embedding 模型会引入不可复现因素）；
2. 它足以复现真实 RAG 的两类典型故障——**召回不到**和**召回到噪声**；
3. 想换真实检索，只需替换 `Retriever.search()` 的实现。

中文没有空格，用字符二元组（bigram）代替分词，是成本最低且效果可接受的做法。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from llm_evals.rag.knowledge import DOCS, Doc

PUNCT = re.compile(r"[\s，。；：！？、,.．;:!?\"'“”‘’（）()\[\]【】\-—~…]+")


def bigrams(text: str) -> set[str]:
    """中文按 2 字切分；英文/数字保留为整体。"""
    clean = PUNCT.sub("", text)
    grams: set[str] = set()
    for i in range(len(clean) - 1):
        grams.add(clean[i : i + 2])
    return {g for g in grams if g.strip()}


@dataclass
class Hit:
    doc: Doc
    score: float


class Retriever:
    def __init__(self, docs: list[Doc] | None = None, top_k: int = 2) -> None:
        self.docs = docs or DOCS
        self.top_k = top_k
        self._df = self._build_df()

    def _build_df(self) -> dict[str, int]:
        df: dict[str, int] = {}
        for d in self.docs:
            for g in bigrams(d.title + d.text):
                df[g] = df.get(g, 0) + 1
        return df

    def _idf(self, gram: str) -> float:
        n = len(self.docs)
        df = self._df.get(gram, 0)
        return math.log((n + 1) / (df + 1)) + 1.0

    def search(self, query: str) -> list[Hit]:
        q = bigrams(query)
        hits: list[Hit] = []
        for d in self.docs:
            dg = bigrams(d.title + d.text)
            if not dg:
                continue
            overlap = q & dg
            if not overlap:
                continue
            # 相似度：加权覆盖 / 查询长度归一
            score = sum(self._idf(g) for g in overlap) / (sum(self._idf(g) for g in q) or 1.0)
            hits.append(Hit(doc=d, score=round(score, 4)))
        hits.sort(key=lambda h: -h.score)
        return hits[: self.top_k]
