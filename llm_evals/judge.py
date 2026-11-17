"""判分器：规则判分优先，语义判分必须可校准。

三条原则：
1. 能用规则判的（包含/排除/正则），绝不交给模型判 —— 零成本、可复现；
2. 必须用模型判时，先用人工标注做校准，算 Cohen's Kappa，低于 0.6 不可用；
3. 判官本身也有版本，改判官等于改尺子，历史分数不能直接比。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Verdict:
    passed: bool
    score: float
    reasons: list[str]


def rule_judge(answer: str, expected_contains: list[str], must_not_contain: list[str]) -> Verdict:
    """规则判分：全部必含项出现、且无禁止项出现，才算通过。"""
    reasons: list[str] = []
    for kw in expected_contains:
        if kw not in answer:
            reasons.append(f"缺少必含内容「{kw}」")
    for kw in must_not_contain:
        if kw in answer:
            reasons.append(f"出现禁止内容「{kw}」")
    return Verdict(passed=not reasons, score=0.0 if reasons else 1.0, reasons=reasons)


def keyword_judge(answer: str, key_terms: list[str], threshold: float = 0.6) -> Verdict:
    """语义判分的可解释近似实现。

    真实生产环境应替换为 LLM 判官（prompt 见 judge_prompt.md），
    本实现用关键词覆盖率近似，好处是完全离线、结果确定、便于做单元测试。
    """
    if not key_terms:
        return Verdict(True, 1.0, [])
    hit = [k for k in key_terms if k in answer]
    score = len(hit) / len(key_terms)
    passed = score >= threshold
    reasons = [] if passed else [f"关键信息覆盖不足：命中 {hit or '无'} / 需要 {key_terms}"]
    return Verdict(passed=passed, score=round(score, 2), reasons=reasons)


def cohen_kappa(y_human: list[int], y_judge: list[int]) -> tuple[float, float, float]:
    """计算 Kappa，返回 (kappa, 观察一致率 po, 期望一致率 pe)。"""
    n = len(y_human)
    if n == 0 or n != len(y_judge):
        return 0.0, 0.0, 0.0
    po = sum(1 for a, b in zip(y_human, y_judge) if a == b) / n
    p_h1 = sum(y_human) / n
    p_j1 = sum(y_judge) / n
    pe = p_h1 * p_j1 + (1 - p_h1) * (1 - p_j1)
    kappa = (po - pe) / (1 - pe) if pe != 1 else 0.0
    return round(kappa, 3), round(po, 3), round(pe, 3)


def calibrate(labels_path: str | Path) -> dict:
    """用人工标注校准判官：判官同意率 + Kappa + 分歧明细。"""
    data = yaml.safe_load(Path(labels_path).read_text(encoding="utf-8"))
    rows = data["samples"]

    human: list[int] = []
    judge: list[int] = []
    disagreements: list[dict] = []

    for r in rows:
        verdict = keyword_judge(r["answer"], r["key_terms"], threshold=data.get("threshold", 0.6))
        j = 1 if verdict.passed else 0
        h = int(r["human"])
        human.append(h)
        judge.append(j)
        if h != j:
            disagreements.append(
                {
                    "id": r["id"],
                    "human": h,
                    "judge": j,
                    "score": verdict.score,
                    "answer": r["answer"],
                    "reasons": verdict.reasons,
                }
            )

    kappa, po, pe = cohen_kappa(human, judge)
    return {
        "n": len(rows),
        "agreement": po,
        "expected_agreement": pe,
        "kappa": kappa,
        "usable": kappa >= data.get("min_kappa", 0.6),
        "disagreements": disagreements,
    }
