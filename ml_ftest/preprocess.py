#!/usr/bin/env python3
"""预处理一致性（测试计划阶段③）—— 线上事故第一名：train-serving skew。

测什么：
    同一段原始信号，分别走"训练侧预处理"和"线上侧预处理"，比对特征与预测。
    只要两套实现有任何差异，就是缺陷 —— 与模型好坏无关。

五种典型 skew：
    ① 单位不一致     训练 g / 线上 m/s²（×9.81）
    ② 滤波不一致     训练带通 0.5–15Hz / 线上不滤
    ③ 窗口不一致     训练 5s 窗 / 线上 3s 窗
    ④ 统计量不一致   训练用训练集全局 mu/sigma / 线上用当前批次自己的
    ⑤ 完全一致       （对照组，验证双路径本身没写错）

这个脚本真正想说的结论：
    **准确率是迟钝的指标。** 单位错 9.81 倍时准确率可能只掉 1 个点，
    但特征已经偏出去几十个标准差 —— 靠"指标没掉"判断预处理一致，是不可靠的。

    python ml_ftest/preprocess.py
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_ftest.metamorphic import LABELS, NearestCentroid  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "ml_preprocess_report.md"

FS = 50
SECONDS = 10
WAVE = {"静止": (0.02, 0.4), "走路": (0.35, 1.8), "跑步": (0.95, 2.8), "骑行": (0.25, 1.5)}
FEATURE_NAMES = ["mean_abs", "std", "zcr", "dominant_freq", "spectral_energy"]


# ---------------- 造数据 ----------------
def make_wave(label: str, npr: np.random.Generator) -> np.ndarray:
    amp, freq = WAVE[label]
    amp *= 1 + npr.normal(0, 0.15)
    freq *= 1 + npr.normal(0, 0.10)
    phase = npr.uniform(0, 2 * np.pi)
    t = np.arange(FS * SECONDS) / FS
    sig = amp * np.sin(2 * np.pi * freq * t + phase)
    return sig + amp * 0.15 * npr.normal(size=len(t))


def build_records(n_subject: int = 60, per_subject: int = 4, seed: int = 20261124):
    rng = random.Random(seed)
    npr = np.random.default_rng(seed)
    weights = [0.25, 0.35, 0.25, 0.15]
    recs = []
    for s in range(n_subject):
        sid = f"S{s + 1:03d}"
        for k in range(per_subject):
            label = rng.choices(LABELS, weights=weights)[0]
            recs.append({"id": f"{sid}-{k}", "subject": sid, "label": label,
                         "sig": make_wave(label, npr)})
    return recs


# ---------------- 预处理组件 ----------------
def bandpass(x: np.ndarray, lo: float = 0.5, hi: float = 15.0) -> np.ndarray:
    """FFT 带通：保留人体运动频段，去掉直流漂移与高频噪声。"""
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / FS)
    spec[(freqs < lo) | (freqs > hi)] = 0.0
    return np.fft.irfft(spec, n=len(x))


def windows(x: np.ndarray, win_sec: float, overlap: float = 0.5) -> list[np.ndarray]:
    n = int(win_sec * FS)
    step = max(1, int(n * (1 - overlap)))
    return [x[i:i + n] for i in range(0, len(x) - n + 1, step)]


def features(w: np.ndarray) -> np.ndarray:
    """单个窗的 5 维特征（与 metamorphic 模块保持一致）。"""
    w = w - w.mean()
    mean_abs = float(np.abs(w).mean())
    std = float(w.std())
    zcr = float(np.mean(np.abs(np.diff(np.signbit(w).astype(int)))))
    spec = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(len(w), 1 / FS)
    dom = float(freqs[1 + int(np.argmax(spec[1:]))]) if len(spec) > 1 else 0.0
    energy = float((spec ** 2).sum() / len(w))
    return np.array([mean_abs, std, zcr, dom, energy])


def preprocess(sig: np.ndarray, cfg: dict) -> np.ndarray:
    """一条原始记录 → 若干窗的特征矩阵（每行一个窗）。"""
    x = np.asarray(sig, dtype=float)
    if cfg.get("unit") == "ms2":
        x = x * 9.81
    if cfg.get("filter", True):
        x = bandpass(x)
    return np.array([features(w) for w in windows(x, cfg.get("window", 5.0))])


BASELINE = {"unit": "g", "filter": True, "window": 5.0}


# ---------------- 模型 ----------------
class Scaler:
    """标准 z-score；统计量来源可控 —— 这正是 skew ④ 的入口。"""

    def fit(self, X: np.ndarray) -> "Scaler":
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-9
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sigma

    def fit_transform_batch(self, X: np.ndarray) -> np.ndarray:
        """错法：用当前批次自己的统计量做标准化（线上常见写法）。"""
        return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)


def record_predict(model, feats_per_rec: list[np.ndarray]) -> list[str]:
    """一条记录有多个窗 → 多数投票，得到记录级预测。"""
    out = []
    for fm in feats_per_rec:
        preds = model.predict(fm)
        out.append(Counter(preds).most_common(1)[0][0])
    return out


def run() -> dict:
    recs = build_records()
    subjects = sorted({r["subject"] for r in recs})
    random.Random(42).shuffle(subjects)
    cut = int(len(subjects) * 0.7)
    train_s = set(subjects[:cut])
    train = [r for r in recs if r["subject"] in train_s]
    test = [r for r in recs if r["subject"] not in train_s]

    # 训练侧：固定用 BASELINE 预处理，统计量来自训练集
    train_feats = [preprocess(r["sig"], BASELINE) for r in train]
    flat_train = np.vstack(train_feats)
    # 标签必须按「窗」展开：一条记录有 3 个窗，就有 3 个样本
    flat_labels = [r["label"] for r, f in zip(train, train_feats) for _ in range(len(f))]
    assert len(flat_labels) == len(flat_train), "标签与特征行数必须一致"

    scaler = Scaler().fit(flat_train)
    model = NearestCentroid().fit(scaler.transform(flat_train), flat_labels)

    # 测试侧：baseline 路径
    base_feats = [preprocess(r["sig"], BASELINE) for r in test]
    base_pred = record_predict(model, [scaler.transform(f) for f in base_feats])
    base_acc = float(np.mean([p == r["label"] for p, r in zip(base_pred, test)]))

    # 每条记录的"标准化后特征均值"——用于算漂移（不受窗数影响）
    def z_mean(fm: np.ndarray) -> np.ndarray:
        return scaler.transform(fm).mean(axis=0)

    base_z = [z_mean(f) for f in base_feats]

    skews = [
        ("⑤ 完全一致（对照）", {}),
        ("① 单位 g → m/s²", {"unit": "ms2"}),
        ("② 带通滤波 → 不滤", {"filter": False}),
        ("③ 窗口 5s → 3s", {"window": 3.0}),
    ]

    rows = []
    for name, override in skews:
        cfg = {**BASELINE, **override}
        feats = [preprocess(r["sig"], cfg) for r in test]
        pred = record_predict(model, [scaler.transform(f) for f in feats])
        acc = float(np.mean([p == r["label"] for p, r in zip(pred, test)]))
        agree = float(np.mean([a == b for a, b in zip(pred, base_pred)]))
        drift = float(np.mean([np.linalg.norm(a - b) for a, b in zip(base_z, [z_mean(f) for f in feats])]))
        rows.append({"name": name, "acc": acc, "delta": acc - base_acc,
                     "agree": agree, "drift": drift})

    # skew ④：统计量改用当前批次（这是归一化阶段，不是预处理阶段）
    flat_test = np.vstack(base_feats)
    feats_batch = [Scaler().fit_transform_batch(f) for f in base_feats]
    pred4 = record_predict(model, feats_batch)
    acc4 = float(np.mean([p == r["label"] for p, r in zip(pred4, test)]))
    rows.append({
        "name": "④ 统计量用当前批次",
        "acc": acc4,
        "delta": acc4 - base_acc,
        "agree": float(np.mean([a == b for a, b in zip(pred4, base_pred)])),
        "drift": float(np.mean([
            np.linalg.norm(a - b) for a, b in
            zip(base_z, [Scaler().fit_transform_batch(f).mean(axis=0) for f in base_feats])
        ])),
    })

    return {
        "n_train_rec": len(train), "n_test_rec": len(test),
        "n_train_win": len(flat_train), "base_acc": base_acc,
        "rows": rows, "flat_test_windows": len(flat_test),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="预处理一致性（train-serving skew）")
    parser.add_argument("--out", default=str(REPORT))
    args = parser.parse_args()

    s = run()
    rows = s["rows"]
    base_acc = s["base_acc"]
    worst = min(rows, key=lambda r: r["acc"])
    blind = next((r for r in rows if r["name"].startswith("①")), rows[0])

    lines = [
        "# 预处理一致性报告（train-serving skew）",
        "",
        "## 一、结论",
        "",
        f"- 记录 **{s['n_train_rec'] + s['n_test_rec']}** 条（训练 {s['n_train_rec']} / 测试 {s['n_test_rec']}，按受试者分组划分）",
        f"- 训练侧窗口样本 **{s['n_train_win']}** 个（5s 窗、50% 重叠）",
        f"- 基线准确率 **{base_acc:.1%}**（两侧预处理完全一致）",
        f"- 五种 skew 中**最严重的一种掉了 {abs(worst['delta']) * 100:.1f} 个百分点**（{worst['name']}）",
        "",
        "| 线上侧实现 | 准确率 | Δ 准确率 | 记录级一致率 | 特征漂移（σ） |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        flag = "" if abs(r["delta"]) < 0.02 else " ⚠"
        lines.append(
            f"| {r['name']} | {r['acc']:.1%} | {r['delta'] * 100:+.1f} pt | "
            f"{r['agree']:.0%} | {r['drift']:.1f}{flag} |"
        )

    win_row = next(r for r in rows if r["name"].startswith("③"))
    unit_row = next(r for r in rows if r["name"].startswith("①"))
    stat_row = next(r for r in rows if r["name"].startswith("④"))

    lines += [
        "",
        "> 判据：**双路径特征必须逐值相等**（浮点容差 1e-9），否则即为缺陷。",
        "> 准确率只用来排序严重程度，**不能用来判断是否存在不一致** —— 理由见第二节。",
        "",
        "## 二、准确率会骗人：看「窗口」那一行",
        "",
        f"窗口 5s → 3s：准确率 {win_row['acc']:.1%}，只比基线低 {abs(win_row['delta']) * 100:.1f} 个百分点，"
        f"记录级一致率 {win_row['agree']:.0%}。",
        "在真实项目里，这种幅度通常被当成「波动」直接放过。",
        "",
        f"但它意味着 **{(1 - win_row['agree']) * 100:.0f}% 的请求给出了不同的分类结果**。"
        "按日请求 100 万算就是几万次差异，而且是**系统性的**（短窗输入会稳定偏），不是随机噪声。",
        "",
        "再看相反的两种情形，更能说明「汇总指标不能当判据」：",
        "",
        "| skew | 特征漂移 | Δ 准确率 | 为什么不是正比 |",
        "|---|---|---|---|",
        f"| 单位 ×9.81 | {unit_row['drift']:.1f}σ | {unit_row['delta'] * 100:+.1f} pt | "
        "整体线性放大，分类器仍能分开，漂移巨大但伤害有限 |",
        f"| 统计量用当前批次 | {stat_row['drift']:.1f}σ | {stat_row['delta'] * 100:+.1f} pt | "
        "漂移不大，但类间结构被压扁，直接崩 |",
        "",
        "**漂移幅度代表「差多远」，不代表「多致命」。** 所以两个指标都只能用来排序，",
        "唯一可靠的判据是：同一输入，两条路径的特征逐值相等。",
        "",
        "## 三、每种 skew 破坏了什么",
        "",
        "| skew | 被破坏的环节 | 受影响的特征 | 为什么准确率会/不会掉 |",
        "|---|---|---|---|",
        "| 单位不一致 | 数据接入层 | mean_abs / std / energy | 线性缩放，分类器可能仍能分开 |",
        "| 滤波不一致 | 信号处理层 | zcr / energy / dominant_freq | 高频噪声直接进特征，过零率暴涨 |",
        "| 窗口不一致 | 分窗层 | dominant_freq（频率分辨率变化） | 窗越短频率分辨率越粗，主频估偏 |",
        "| 统计量不一致 | 归一化层 | 全部 5 维 | 分布整体平移缩放，质心距离失真 |",
        "",
        "**四层分别对应四个地方**：数据接入、信号处理、分窗、归一化。",
        "排查顺序就按这个来 —— 从最上游往下游查，因为上游错会污染下游全部特征。",
        "",
        "## 四、双路径一致性测试怎么写（可直接用）",
        "",
        "```python",
        "def test_train_serving_consistency():",
        '    """同一段原始数据，两条路径的输出必须逐值相等。"""',
        "    sig = load_one_raw_sample()",
        "",
        "    train_feat = preprocess_train(sig)   # 训练侧实现",
        "    serve_feat = preprocess_serve(sig)   # 线上侧实现（可以是另一个文件/服务）",
        "",
        "    assert train_feat.shape == serve_feat.shape",
        "    np.testing.assert_allclose(train_feat, serve_feat, rtol=0, atol=1e-9)",
        "```",
        "",
        "三个要点：",
        "",
        "1. **必须用同一段原始数据**，不能各自造数据 —— 否则比的是数据差异，不是实现差异；",
        "2. **`rtol=0`**：只允许极小的绝对误差，不允许相对误差（幅度小时相对误差会放过真问题）；",
        "3. **覆盖边界输入**：全零、饱和削顶、超短（不足一个窗）、采样率异常。"
        "预处理 bug 几乎都只在边界输入上暴露。",
        "",
        "## 五、局限（要主动说）",
        "",
        "1. 训练侧的「正确实现」是**本脚本自己写的**，真实项目里应以训练代码为准，"
        "这里只是演示比对方法；",
        "2. 用的是最近质心分类器，对特征偏移不敏感；换真实模型（CNN/LSTM）掉点可能更剧烈，"
        "**所以这里的 Δ 是下限，不是上限**；",
        "3. 只有单轴信号，没有覆盖多轴错位、坐标系旋转这类更隐蔽的 skew；",
        "4. 数据是合成的，真实设备还有固件版本、量程切换、采样抖动等差异，skew 来源更多。",
        "",
        "## 六、复现",
        "",
        "```bash",
        "python ml_ftest/preprocess.py",
        "```",
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"记录 {s['n_train_rec'] + s['n_test_rec']} 条，基线准确率 {base_acc:.1%}")
    for r in rows:
        print(f"  {r['name']:<16} 准确率 {r['acc']:.1%}  Δ{r['delta'] * 100:+.1f}pt  "
              f"一致率 {r['agree']:.0%}  漂移 {r['drift']:.1f}σ")
    print(f"报告：{out}")


if __name__ == "__main__":
    main()
