"""图表输出：所有图统一风格，直接进报告。

注意：图内文字用英文，避免服务器上缺中文字体导致方块乱码
（这是很多人踩过的坑，报告发出去全是"□□□"）。
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # 无界面环境必须，否则 CI 上会报错
import matplotlib.pyplot as plt
import pandas as pd

from analysis.metrics import CAPACITY_PASS_RATIO, TEMP_PASS_LIMIT_C

STYLE = {
    "figure.dpi": 150,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}
plt.rcParams.update(STYLE)


def plot_capacity_fade(df: pd.DataFrame, out_dir: str) -> str:
    """容量衰减曲线：直观看出哪台设备先掉到 80% 以下。"""
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for sn, g in df.groupby("sn"):
        g = g.sort_values("cycle")
        base = g["capacity_mah"].iloc[0]
        ax.plot(g["cycle"], g["capacity_mah"] / base * 100, label=sn, linewidth=1.4)

    ax.axhline(CAPACITY_PASS_RATIO * 100, color="#c0392b", linestyle="--", linewidth=1)
    ax.text(0, CAPACITY_PASS_RATIO * 100 + 1.2, "pass limit 80%", color="#c0392b", fontsize=8)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity retention (%)")
    ax.set_title("Battery capacity fade")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()

    path = os.path.join(out_dir, "capacity_fade.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_temp_distribution(df: pd.DataFrame, out_dir: str) -> str:
    """温升分布箱线图：看离散度，比只看平均值更能发现问题。"""
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    groups = [g["temp_c"].values for _, g in df.groupby("sn")]
    labels = [sn for sn, _ in df.groupby("sn")]
    ax.boxplot(groups, widths=0.5)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.axhline(TEMP_PASS_LIMIT_C, color="#c0392b", linestyle="--", linewidth=1)
    ax.text(1.0, TEMP_PASS_LIMIT_C + 0.8, "limit 45C", color="#c0392b", fontsize=8)
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Temperature distribution by device")
    plt.setp(ax.get_xticklabels(), rotation=20, fontsize=7)
    fig.tight_layout()

    path = os.path.join(out_dir, "temp_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_pass_rate(metrics: pd.DataFrame, out_dir: str) -> str:
    """各设备判定结果柱状图，一眼看出谁没通过、卡在哪一项。"""
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    x = range(len(metrics))
    ax.bar([i - 0.2 for i in x], metrics["retention_pct"], width=0.4, label="retention %")
    ax.bar([i + 0.2 for i in x], metrics["p95_temp_c"] * 2, width=0.4, label="P95 temp x2 (scaled)")

    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics["sn"], rotation=20, fontsize=7)
    ax.set_title("Retention vs P95 temperature (scaled)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()

    path = os.path.join(out_dir, "pass_overview.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def make_all(df: pd.DataFrame, metrics: pd.DataFrame, out_dir: str) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    return {
        "capacity": plot_capacity_fade(df, out_dir),
        "temp": plot_temp_distribution(df, out_dir),
        "pass": plot_pass_rate(metrics, out_dir),
    }
