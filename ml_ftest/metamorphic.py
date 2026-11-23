#!/usr/bin/env python3
"""蜕变测试（测试计划阶段④）。

为什么需要先造特征和模型：
    蜕变测试的本质是"改输入 → 重新预测 → 看输出是否按预期关系变化"。
    只拿预测结果 CSV 做不到这件事，必须有一个能重新推理的模型。

这里用一个 **最近质心分类器**（纯 numpy，20 行）：
    训练集算每类特征均值做质心，预测时取最近质心。
    好处是可解释、无依赖、结果确定 —— 测试框架自己也要可信。

五条蜕变关系（前三条要求"不变"，后两条是"反向关系"）：
    MR-1 幅度缩放 ±10%（戴松/戴紧）     → 类别不变
    MR-2 特征加噪 5%                     → 类别不变
    MR-3 主频微偏移 ±5%（佩戴角度变化）  → 类别不变
    MR-4 单个特征缺失（zcr 置零）        → 类别不变率应下降，暴露单特征依赖
    MR-5 全零输入（无运动）              → **必须**判为静止，不能判成跑步

    python ml_ftest/metamorphic.py
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ml" / "feature_sample.csv"
REPORT = ROOT / "reports" / "ml_metamorphic_report.md"

LABELS = ["静止", "走路", "跑步", "骑行"]
FEATURES = ["mean_acc", "std_acc", "zcr", "dominant_freq", "spectral_energy"]

# 每个类别的特征基准值（有物理含义：幅度、波动、过零率、主频、能量）
BASE = {
    "静止": [0.02, 0.010, 2.0, 0.5, 0.10],
    "走路": [0.35, 0.180, 45.0, 2.2, 2.50],
    "跑步": [0.95, 0.420, 95.0, 3.2, 8.00],
    "骑行": [0.25, 0.120, 30.0, 1.6, 1.80],
}
# 受试者效应：不同人佩戴姿势/体态不同，整体偏移
SUBJECT_JITTER = [0.08, 0.10, 0.12, 0.10, 0.08]
NOISE = 0.06


def gen_features(n: int = 600, seed: int = 20261123) -> tuple[np.ndarray, list[str], list[str]]:
    rng = random.Random(seed)
    npr = np.random.default_rng(seed)
    X, y, subjects = [], [], []
    for _ in range(n):
        label = rng.choices(LABELS, weights=[0.25, 0.35, 0.25, 0.15])[0]
        subj = f"S{rng.randint(1, 40):03d}"
        base = np.array(BASE[label], dtype=float)
        jitter = 1.0 + npr.normal(0, 0.08)  # 受试者整体缩放
        x = base * jitter * (1 + npr.normal(0, NOISE, size=len(base)))
        X.append(x)
        y.append(label)
        subjects.append(subj)
    return np.array(X), y, subjects


class NearestCentroid:
    """最近质心分类器：先按训练集做 z-score，再比欧氏距离。"""

    def fit(self, X: np.ndarray, y: list[str]) -> "NearestCentroid":
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-9
        Z = (X - self.mu) / self.sigma
        self.centroids = {c: Z[[i for i, t in enumerate(y) if t == c]].mean(axis=0) for c in set(y)}
        return self

    def predict(self, X: np.ndarray) -> list[str]:
        Z = (X - self.mu) / self.sigma
        out = []
        for row in Z:
            best, best_d = None, float("inf")
            for c, cen in self.centroids.items():
                d = float(np.linalg.norm(row - cen))
                if d < best_d:
                    best, best_d = c, d
            out.append(best)
        return out


# ---------------- 蜕变关系 ----------------
def mr_scale(X: np.ndarray, factor: float) -> np.ndarray:
    """幅度类特征整体缩放（mean_acc / std_acc / spectral_energy）。"""
    Y = X.copy()
    for i, name in enumerate(FEATURES):
        if name in ("mean_acc", "std_acc", "spectral_energy"):
            Y[:, i] *= factor
    return Y


def mr_noise(X: np.ndarray, ratio: float, seed: int = 7) -> np.ndarray:
    npr = np.random.default_rng(seed)
    return X * (1 + npr.normal(0, ratio, size=X.shape))


def mr_freq_shift(X: np.ndarray, factor: float) -> np.ndarray:
    Y = X.copy()
    Y[:, FEATURES.index("dominant_freq")] *= factor
    return Y


def mr_drop_feature(X: np.ndarray, name: str = "zcr") -> np.ndarray:
    Y = X.copy()
    Y[:, FEATURES.index(name)] = 0.0
    return Y


def mr_zero(X: np.ndarray) -> np.ndarray:
    return np.zeros_like(X)


def render(X, y, subjects, report_path: Path) -> dict:
    # 按受试者分组划分 70/30（与阶段①结论一致：绝不用随机划分）
    subj = sorted(set(subjects))
    random.Random(42).shuffle(subj)
    cut = int(len(subj) * 0.7)
    train_s = set(subj[:cut])
    test_idx = [i for i, s in enumerate(subjects) if s not in train_s]
    train_idx = [i for i, s in enumerate(subjects) if s in train_s]

    model = NearestCentroid().fit(X[train_idx], [y[i] for i in train_idx])
    Xt = X[test_idx]
    yt = [y[i] for i in test_idx]
    base_pred = model.predict(Xt)
    base_acc = float(np.mean([p == t for p, t in zip(base_pred, yt)]))

    def invariance(pred_a: list[str], pred_b: list[str]) -> float:
        return float(np.mean([a == b for a, b in zip(pred_a, pred_b)]))

    results = []

    for factor in (1.10, 0.90):
        p = model.predict(mr_scale(Xt, factor))
        results.append((f"MR-1 幅度缩放 ×{factor:.2f}", "类别不变", invariance(base_pred, p), 0.90))
    p = model.predict(mr_noise(Xt, 0.05))
    results.append(("MR-2 特征加噪 5%", "类别不变", invariance(base_pred, p), 0.90))
    for factor in (1.05, 0.95):
        p = model.predict(mr_freq_shift(Xt, factor))
        results.append((f"MR-3 主频偏移 ×{factor:.2f}", "类别不变", invariance(base_pred, p), 0.90))
    p = model.predict(mr_drop_feature(Xt, "zcr"))
    results.append(("MR-4 过零率缺失", "类别不变（允许下降）", invariance(base_pred, p), 0.75))

    # MR-5 反向关系：全零必须判静止
    p_zero = model.predict(mr_zero(Xt[:1]))
    mr5_ok = p_zero[0] == "静止"
    results.append(("MR-5 全零输入", "必须判为静止", 1.0 if mr5_ok else 0.0, 1.0))

    passed = [r for r in results if r[2] >= r[3]]
    failed = [r for r in results if r[2] < r[3]]

    lines = [
        "# 蜕变测试报告",
        "",
        "## 一、结论",
        "",
        f"- 训练/测试按**受试者分组**划分：训练 {len(train_idx)} 条、测试 {len(test_idx)} 条（受试者无交叉）",
        f"- 基线准确率 **{base_acc:.1%}**（最近质心分类器）",
        f"- 蜕变关系 **{len(results)}** 条，通过 **{len(passed)}** 条"
        + (f"，未通过 **{len(failed)}** 条" if failed else ""),
        "",
        "| 蜕变关系 | 期望关系 | 实测不变率 | 阈值 | 结论 |",
        "|---|---|---|---|---|",
    ]
    for name, rel, val, th in results:
        mark = "通过" if val >= th else "**未通过**"
        lines.append(f"| {name} | {rel} | {val:.1%} | {th:.0%} | {mark} |")

    lines += [
        "",
        "## 二、怎么解读",
        "",
        "- **MR-1/2/3 都要求「类别不变」**：如果加 5% 噪声就判错，说明模型学的是噪声而不是动作本身；",
        "- **MR-4 阈值放低到 75%**：单个特征缺失必然带来下降，这里看的不是「能不能过」，"
        "而是**下降幅度是否可接受**（下降过多说明过度依赖单一特征）；",
        "- **MR-5 是反向关系**：不要求「不变」，而要求「必须变成静止」。"
        "全零输入被判成「跑步」是典型的模型没学到物理规律。",
        "",
        "## 三、与阶段②的联动",
        "",
        "蜕变失败的样本要回到切片分析里看：**如果失败集中在某一切片**（如低采样率、高噪声），",
        "说明根因是数据覆盖不足，而不是「鲁棒性差」这种笼统结论。",
        "",
        "## 四、局限（要主动说）",
        "",
        "1. 这里的蜕变关系作用在**已提取的特征**上，不是原始信号。"
        "真实项目应在信号层施加变换（加噪、重采样、截断），更接近线上实际情况；",
        "2. 分类器是最近质心，能力弱于真实模型，**不变率只能横向比较，不能当成产品指标**；",
        "3. 阈值（90%/75%）是按经验定的，换业务要重定——**阈值本身要有书面理由**。",
        "",
        "## 五、复现",
        "",
        "```bash",
        "python ml_ftest/metamorphic.py",
        "```",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"baseline_acc": base_acc, "results": results, "n_test": len(test_idx)}


def main() -> None:
    parser = argparse.ArgumentParser(description="蜕变测试")
    parser.add_argument("--out", default=str(REPORT))
    args = parser.parse_args()

    X, y, subjects = gen_features()
    stats = render(X, y, subjects, Path(args.out))

    print(f"测试样本 {stats['n_test']} 条，基线准确率 {stats['baseline_acc']:.1%}")
    for name, _, val, th in stats["results"]:
        print(f"  {name}: {val:.1%}（阈值 {th:.0%}）{'✓' if val >= th else '✗'}")
    print(f"报告：{args.out}")


if __name__ == "__main__":
    main()
