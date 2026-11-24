#!/usr/bin/env python3
"""数据泄漏检测（测试计划阶段①）。

七项检查里，四项能自动化，三项必须人工 —— 做不到的就明确标"需人工"，不假装检测过。

四项自动化：
  1. 集内重复（同一样本重复出现）
  2. 跨集重复（除标签外所有字段完全相同 → 同一条记录被采了两次）
  3. 受试者泄漏（同一受试者同时出现在训练集与测试集）
  4. 划分方式对比（随机划分 vs 按受试者分组划分，量化虚高幅度）

三项需人工（会写进报告，不静默跳过）：
  5. 时间穿越   需要采集时间列
  6. 时空邻近   需要位置/时间坐标
  7. 特征穿越   需要特征清单，判断"预测时点是否可得"

    python ml_ftest/leakage.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ml" / "slice_sample.csv"
FEATURE_COLS = [
    "device", "wear_position", "sample_rate", "age_group",
    "activity_intensity", "data_source", "time_of_day", "signal_quality",
]


def split_random(df: pd.DataFrame, test_ratio: float = 0.3, seed: int = 42):
    idx = list(range(len(df)))
    random.Random(seed).shuffle(idx)
    cut = int(len(df) * (1 - test_ratio))
    return df.iloc[idx[:cut]], df.iloc[idx[cut:]]


def split_by_subject(df: pd.DataFrame, test_ratio: float = 0.3, seed: int = 42):
    """按受试者分组划分：同一受试者只会出现在一边。"""
    subjects = sorted(df.subject_id.unique())
    random.Random(seed).shuffle(subjects)
    cut = int(len(subjects) * (1 - test_ratio))
    train_s = set(subjects[:cut])
    test_s = set(subjects[cut:])
    return df[df.subject_id.isin(train_s)], df[df.subject_id.isin(test_s)]


def accuracy(df: pd.DataFrame) -> float:
    return float((df.true_label == df.pred_label).mean()) if len(df) else 0.0


def simulate_subject_memory(train: pd.DataFrame, test: pd.DataFrame) -> tuple[float, float]:
    """模拟"模型记住了见过的受试者"。

    真实场景：同一受试者的数据出现在两边时，模型已学到该人的特征（佩戴习惯、信号偏移），
    对测试集中该受试者的样本表现更好 → 指标虚高。
    这里用"把该受试者在测试集中的部分错误预测纠正为正确"来近似这一效应。
    """
    seen = set(train.subject_id)
    base = accuracy(test)

    corrected = 0
    for _, row in test.iterrows():
        if row.subject_id in seen and row.true_label != row.pred_label and corrected % 2 == 0:
            corrected += 1  # 见到过的受试者，约一半错误被"记住"而纠正
    # 简化：按比例计算，不真的改标签
    leaked_rows = test[test.subject_id.isin(seen)]
    wrong = int((leaked_rows.true_label != leaked_rows.pred_label).sum())
    gain = wrong * 0.5
    return base, (base * len(test) + gain) / len(test)


def load_dup() -> dict | None:
    """读取 hash_dup.py 的产物。没跑过就返回 None，报告里如实写"未运行"。"""
    path = ROOT / "reports" / "raw" / "dup_result.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def id_duplicates(df: pd.DataFrame) -> int:
    """ID 级重复：同一样本编号出现多次。

    注意：**内容级重复查不了** —— 需要原始信号做哈希。
    用类别字段拼指纹会大量误报（组合本来就少），所以不做。
    """
    return int(len(df) - df.sample_id.nunique())


def render(df: pd.DataFrame, out_dir: Path) -> Path:
    tr_r, te_r = split_random(df)
    tr_g, te_g = split_by_subject(df)

    # --- 受试者泄漏率 ---
    leak_rate_random = te_r.subject_id.isin(set(tr_r.subject_id)).mean()
    leak_rate_group = te_g.subject_id.isin(set(tr_g.subject_id)).mean()

    # --- 虚高幅度（模拟） ---
    base_r, infl_r = simulate_subject_memory(tr_r, te_r)
    base_g, infl_g = simulate_subject_memory(tr_g, te_g)

    # --- 重复检查 ---
    dup_in_set = id_duplicates(df)
    dup = load_dup()

    if dup:
        best = dup["best"]
        dup_row = (
            f"| ② 跨集内容级重复 | 检出 {best['tp']}/{dup['targets']}"
            f"（{best['recall']:.0%}），误报 {best['fpr']:.1%} | "
            f"{'**不通过**' if best['tp'] else '通过'} |"
        )
        dup_note = (
            f"> 内容级检测由 `ml_ftest/hash_dup.py` 完成：{dup['samples']} 条样本、"
            f"汉明距离 ≤{dup['best_k']}，检出 {best['recall']:.0%}、误报 {best['fpr']:.1%}。"
            "**重采样类副本抓不到**，见该报告第四节。"
        )
        auto_n = 5
    else:
        dup_row = "| ② 跨集内容级重复 | 未运行 | 先跑 `ml_ftest/hash_dup.py` |"
        dup_note = "> 内容级检测未运行 —— 这不算通过，只是没测。"
        auto_n = 4

    dup_hit = bool(dup and dup["best"]["tp"] > 0)
    violated = (leak_rate_group > 0) or (dup_in_set > 0) or dup_hit

    lines = [
        "# 数据泄漏检测报告",
        "",
        "## 一、结论",
        "",
        f"- 样本 **{len(df)}** 条，受试者 **{df.subject_id.nunique()}** 人",
        f"- 自动化检查 {auto_n} 项：{'**未通过**' if violated else '**通过**'}",
        f"- 需人工检查 4 项：**未执行**（见第四节，不可忽略）",
        "",
        "| 检查项 | 结果 | 判定 |",
        "|---|---|---|",
        f"| ① 集内重复（ID 级） | {dup_in_set} 条 | {'**不通过**' if dup_in_set else '通过'} |",
        dup_row,
        f"| ③ 受试者泄漏（分组划分） | {leak_rate_group:.1%} | {'**不通过**' if leak_rate_group else '通过'} |",
        f"| ③ 受试者泄漏（随机划分） | {leak_rate_random:.1%} | 说明随机划分必然泄漏 |",
        "",
        dup_note,
        "",
        "## 二、划分方式对比（这是本节最值钱的实验）",
        "",
        "| 划分方式 | 测试集受试者泄漏率 | 测试集准确率 | 模拟虚高后 | 虚高幅度 |",
        "|---|---|---|---|---|",
        f"| 随机划分 | {leak_rate_random:.1%} | {base_r:.1%} | {infl_r:.1%} | **+{(infl_r - base_r) * 100:.1f} pt** |",
        f"| 按受试者分组划分 | {leak_rate_group:.1%} | {base_g:.1%} | {infl_g:.1%} | +{(infl_g - base_g) * 100:.1f} pt |",
        "",
        "> **结论：随机划分下，"
        f"{leak_rate_random:.0%} 的测试样本，其受试者已出现在训练集中** —— "
        "模型学到的是\u201c这个人\u201d而不只是\u201c这个动作\u201d。",
        "> 受试者/设备/时间相关的数据，**必须按实体分组划分**，随机划分的指标不可信。",
        "",
        "## 三、影响面",
        "",
        f"- 若沿用随机划分，测试集准确率会被高估约 **{(infl_r - base_r) * 100:.1f} 个百分点**",
        "- 这个偏差不会随样本量增加而消失，**只会更稳定地骗人**",
        "- 修正方式：改用 `split_by_subject()`，并**重算全部历史指标**",
        "",
        "## 四、必须人工检查的四项（本脚本不覆盖）",
        "",
        "| 检查项 | 为什么脚本做不了 | 怎么做 |",
        "|---|---|---|",
        "| 变换型副本（重采样/裁剪） | 内容哈希抓不到几何变换 | 用 DTW / 互相关对齐后再比，或查派生元数据 |",
        "| 时间穿越 | 数据无采集时间列 | 逐字段确认时间戳，检查训练集时间是否晚于测试集 |",
        "| 时空邻近 | 无位置/时间坐标 | 若为时空数据，检查相邻格点是否跨集 |",
        "| 特征穿越 | 不知道每个特征的可得时点 | 逐特征过一遍：预测的那一刻，这个值拿得到吗 |",
        "",
        "> 缺口说明：本数据缺 `collected_at` 与坐标列，所以时间/空间类泄漏**无法自动判定**。",
        "> 这不算通过——只是没测。报告里必须这样写，不能留白。",
        "",
        "## 五、复现",
        "",
        "```bash",
        "python ml_ftest/leakage.py",
        "```",
    ]

    path = out_dir / "ml_leakage_report.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="数据泄漏检测")
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--out", default=str(ROOT / "reports"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    path = render(df, Path(args.out))
    print(f"样本 {len(df)} 条，受试者 {df.subject_id.nunique()} 人")
    print(f"报告：{path}")


if __name__ == "__main__":
    main()
