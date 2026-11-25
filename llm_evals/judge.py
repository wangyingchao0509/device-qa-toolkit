"""判分器：规则判分优先，语义判分必须可校准。

三条原则：
1. 能用规则判的（包含/排除/正则），绝不交给模型判 —— 零成本、可复现；
2. 必须用模型判时，先用人工标注做校准，算 Cohen's Kappa，低于 0.6 不可用；
3. 判官本身也有版本，改判官等于改尺子，历史分数不能直接比；
4. 光有 Kappa 不够，还要给区间 —— 19 条样本的 0.756 和 200 条的 0.756 不是一回事。
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


def bootstrap_kappa_ci(
    y_human: list[int],
    y_judge: list[int],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, list[float]]:
    """用 bootstrap 估 Kappa 的置信区间，返回 (下限, 上限, 全部重采样值)。

    为什么必须 bootstrap：
        Kappa 的分母含 Pe，Pe 又是两个边缘分布的乘积 —— 它没有闭式方差公式，
        硬套比例的 Wald 区间会算错（小样本时上限甚至超过 1）。

    为什么跳过退化样本：
        有放回重 sampling 时可能抽到「全判合格」这样的样本，
        此时 Pe=1、Kappa 无定义，必须丢弃而不是当成 0。
    """
    import random

    n = len(y_human)
    if n < 5 or n != len(y_judge):
        return 0.0, 1.0, []

    rng = random.Random(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        h = [y_human[i] for i in idx]
        j = [y_judge[i] for i in idx]
        if len(set(h)) < 2 or len(set(j)) < 2:
            continue  # 退化样本，Kappa 无定义
        k, _, _ = cohen_kappa(h, j)
        vals.append(k)

    if len(vals) < 50:
        return 0.0, 1.0, vals

    vals.sort()
    lo = vals[int(alpha / 2 * len(vals))]
    hi = vals[int((1 - alpha / 2) * len(vals)) - 1]
    return round(lo, 3), round(hi, 3), vals


def recommend_sample_size(kappa: float, lo: float, hi: float, threshold: float, n: int) -> int | None:
    """要让区间下限站上阈值，样本大约要扩到多少条。

    宽度 ∝ 1/√n，所以 n' = n × (当前宽度 / 目标宽度)²。
    """
    width = hi - lo
    need = 2 * (kappa - threshold)
    if need <= 0:
        return None  # 点估计本身就低于阈值，加样本也救不了，该改判据
    if width <= need:
        return n
    return int(n * (width / need) ** 2) + 1


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
    min_kappa = data.get("min_kappa", 0.6)
    ci_lo, ci_hi, boot = bootstrap_kappa_ci(human, judge)

    # 区间跨过阈值 = 证据不足以断言达标，这是比「Kappa 够不够」更严格的判断
    below = sum(1 for v in boot if v < min_kappa) / len(boot) if boot else 0.0
    if ci_lo >= min_kappa:
        conclusion = "达标（区间下限高于阈值）"
    elif ci_hi < min_kappa:
        conclusion = "不达标（区间上限低于阈值）"
    else:
        conclusion = "初步可用（区间跨阈值，证据不足）"

    return {
        "n": len(rows),
        "agreement": po,
        "expected_agreement": pe,
        "kappa": kappa,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "below_ratio": round(below, 3),
        "recommend_n": recommend_sample_size(kappa, ci_lo, ci_hi, min_kappa, len(rows)),
        "conclusion": conclusion,
        "usable": kappa >= min_kappa,
        "disagreements": disagreements,
    }
