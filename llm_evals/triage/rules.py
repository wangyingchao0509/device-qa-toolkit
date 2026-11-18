"""Bad Case 归因规则：能确定性判的，绝不让模型碰。

设计原则
--------
1. 每条规则必须有"证据"，不能只给结论 —— 缺陷单里写不出证据的归因等于没归因；
2. 证据不足时**明说"判断不了"**，并指出缺了哪项信息（置信度 low）；
3. 规则顺序固定：安全 → 检索 → 生成 → 参数 → 兜底，因为一层的失败会掩盖后一层。

六层归因（与 README 一致）
--------------------------
① 参数层   max_tokens 截断、temperature 异常
② 检索层   该找的资料没检索到 / 没排到前面
③ 上下文层 关键信息被截断或淹没
④ 生成层   幻觉、漏要点、答非所问
⑤ 安全层   注入泄露、越权、PII
⑥ 数据层   知识库里根本没有（与检索层的区别：检索层的文档存在但没召回）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MISSING_RE = re.compile(r"缺少必含内容「([^」]+)」")
FORBIDDEN_RE = re.compile(r"出现禁止内容「([^」]+)」")

SAFETY_INTENTS = {"提示词注入", "越界拒答", "越权拒答", "事实准确"}


@dataclass
class Case:
    id: str
    intent: str
    question: str
    answer: str
    reasons: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    severity: str = "P1"

    @property
    def missing_terms(self) -> list[str]:
        out = []
        for r in self.reasons:
            out += MISSING_RE.findall(r)
        return out

    @property
    def forbidden_terms(self) -> list[str]:
        out = []
        for r in self.reasons:
            out += FORBIDDEN_RE.findall(r)
        return out


@dataclass
class Attribution:
    case_id: str
    layer: str
    verdict: str
    evidence: str
    advice: str
    owner: str
    confidence: str  # high / medium / low
    blocking_gap: str = ""  # 缺什么信息导致判不了
    source: str = "rule"  # rule / llm，用于事后评估"模型补充了多少"


def _ctx_text(case: Case) -> str:
    return "".join(case.contexts)


def attribute(case: Case) -> Attribution:
    answer = (case.answer or "").strip()
    ctx = _ctx_text(case)
    missing = case.missing_terms
    forbidden = case.forbidden_terms

    # ⑤ 安全层：出现禁止内容，且属于高风险意图
    if forbidden and case.intent in SAFETY_INTENTS:
        return Attribution(
            case.id, "安全层", "输出泄露/事实错误",
            f"答案出现禁止内容：{'、'.join(forbidden)}",
            "先修提示词与拒答策略，必要时加输出过滤；该条应作为发版阻断项",
            "提示词/算法", "high",
        )

    # ④ 生成层：出现禁止内容（非安全意图）→ 夹带无依据内容
    if forbidden:
        return Attribution(
            case.id, "生成层", "夹带无依据内容",
            f"答案出现不应出现的表述：{'、'.join(forbidden)}",
            "在提示词中加引用约束（要求逐句给依据），或降低温度后复测",
            "提示词", "high",
        )

    # ② 检索层：关键要点完全不在检索内容里
    if missing and case.contexts:
        absent = [t for t in missing if t not in ctx]
        if absent:
            return Attribution(
                case.id, "检索层", "关键信息未召回",
                f"要点「{'、'.join(absent)}」不在检索到的 {len(case.contexts)} 段内容中",
                "查切分策略与检索 Top-K；若知识库确实没有该信息，则属数据层缺口",
                "检索/数据", "high",
            )
        # 要点在上下文里，但答案没用 → 生成层漏用
        return Attribution(
            case.id, "生成层", "答案缺失要点（资料里有但没用上）",
            f"要点「{'、'.join(missing)}」已存在于检索内容，但未出现在答案中",
            "调整拼接策略（取更多句/换抽取方式），或在提示词中要求覆盖全部要点",
            "提示词/拼接逻辑", "high",
        )

    # ① 参数层：答案极短且**没有句末标点** → 很像被 max_tokens 截断
    #    （"请咨询客服。"这种是主动拒答，不是截断，不能误判成参数问题）
    truncated_like = len(answer) < 12 and (not answer or answer[-1] not in "。！？.!?」")
    if missing and truncated_like:
        return Attribution(
            case.id, "参数层", "疑似输出被截断",
            f"答案仅 {len(answer)} 字却判为缺少「{'、'.join(missing)}」",
            "先核对 max_tokens 与 stop 配置，排除参数问题再报模型缺陷",
            "测试自查", "medium",
        )

    # 无法判断：缺检索内容
    if missing and not case.contexts:
        return Attribution(
            case.id, "生成层", "无法区分：缺少检索内容",
            f"缺少要点「{'、'.join(missing)}」，但结果文件未记录 context",
            "评测结果需带上检索片段（contexts），否则无法区分检索层与生成层",
            "评测流程改造", "low",
            "结果文件缺少 contexts 字段",
        )

    # 兜底 A：风险意图失败，即使缺证据也应优先按安全缺陷排查（中置信）
    if case.intent in SAFETY_INTENTS:
        return Attribution(
            case.id, "安全层", "高风险意图失败（需人工确认失效点）",
            f"分类「{case.intent}」属高风险意图，但结果文件缺少判据原文",
            "按安全缺陷流程排查：先确认是否泄露/越界作答，再核对判据是否本身有误",
            "提示词/人工", "medium",
            "缺少判据原文，无法定位具体失效点",
        )

    # 兜底 B：没有任何可用证据
    return Attribution(
        case.id, "未知", "证据不足，需人工判断",
        "结果文件中没有可解析的判据信息（缺少 missing/forbidden 标记）",
        "补充判据原文（expected_contains / must_not_contain）后再归因",
        "人工", "low",
        "结果文件缺少判据原文",
    )


def summarize(attributions: list[Attribution]) -> dict:
    total = len(attributions)
    by_conf: dict[str, int] = {}
    by_layer: dict[str, int] = {}
    for a in attributions:
        by_conf[a.confidence] = by_conf.get(a.confidence, 0) + 1
        by_layer[a.layer] = by_layer.get(a.layer, 0) + 1
    high = by_conf.get("high", 0)
    medium = by_conf.get("medium", 0)
    covered = high + medium
    return {
        "total": total,
        "by_layer": dict(sorted(by_layer.items(), key=lambda kv: -kv[1])),
        "by_confidence": by_conf,
        "rule_coverage": round(covered / total, 3) if total else 0.0,
        "high_conf_coverage": round(high / total, 3) if total else 0.0,
        "medium_conf": medium,
        "need_model_or_human": total - covered,
    }
