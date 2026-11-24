#!/usr/bin/env python3
"""内容级重复检测（补充阶段①的第 5 项自动化检查）。

为什么需要它：
    ID 不同、内容相同的样本，靠 sample_id 查不出来。
    最常见的来源是**数据增强**——同一段信号加噪 / 缩放后生成新样本，
    如果"先增强再划分"，副本会同时落在训练集和测试集里。

这个脚本逐级对比四种检测方式，一条比一条强：
    1. 精确哈希（原始浮点字节做 sha256）      → 检出 0%
    2. 量化指纹（形状归一化 + 量化 + 精确匹配）→ 检出约 1/3
    3. 二值签名 + 汉明距离                     → 检出约 6 成，误报 0
    4. 逻辑键（受试者 + 采集场次）             → 检出 100%

为什么不是"越高级越好"：第 4 种准确率最高，但它要求**有采集元数据**。
哈希是「只剩信号、什么都不知道」时的兜底手段 —— 报告里会说清这个前提。

    python ml_ftest/hash_dup.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_ftest.metamorphic import LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "ml_dup_report.md"
RAW = ROOT / "reports" / "raw" / "dup_result.json"

FS = 50          # 采样率 Hz
SECONDS = 10     # 每条信号 10 秒
BINS = 128       # 签名维度（分块数）
Q = 0.25         # 量化指纹的格子宽度（单位：标准差）
K_GRID = [0, 1, 2, 4, 6, 8]
TARGET_FPR = 0.01

# 每类动作的幅度与主频（有物理含义）
WAVE = {"静止": (0.02, 0.4), "走路": (0.35, 1.8), "跑步": (0.95, 2.8), "骑行": (0.25, 1.5)}
KIND_CN = {"original": "原件（副本在对侧）", "noise": "加噪 5%", "scale": "幅度 ×1.10", "freq": "重采样 ×1.05"}


# ---------------- 造数据（波形层，不是特征层） ----------------
def _wave(label: str, npr: np.random.Generator) -> np.ndarray:
    """按动作类型造一段 10 秒波形：正弦主分量 + 15% 采集噪声。"""
    amp, freq = WAVE[label]
    amp *= 1 + npr.normal(0, 0.15)     # 受试者间幅度差异
    freq *= 1 + npr.normal(0, 0.10)    # 受试者间步频差异
    phase = npr.uniform(0, 2 * np.pi)  # 每次采集起始相位不同
    t = np.arange(FS * SECONDS) / FS
    sig = amp * np.sin(2 * np.pi * freq * t + phase)
    return sig + amp * 0.15 * npr.normal(size=len(t))


def augment(sig: np.ndarray, rng: random.Random, npr: np.random.Generator):
    """三种增强之一：加噪 5% / 幅度 ×1.10 / 重采样 ×1.05（对应 MR-1/2/3）。"""
    kind = rng.choice(["noise", "scale", "freq"])
    if kind == "noise":
        return sig + np.abs(sig).max() * 0.05 * npr.normal(size=len(sig)), kind
    if kind == "scale":
        return sig * 1.10, kind
    # 频率偏移用重采样模拟：1.05 倍主频 ≈ 把时间轴压缩 1.05
    idx = np.linspace(0, len(sig) - 1, int(len(sig) / 1.05))
    out = np.interp(idx, np.arange(len(sig)), sig)
    return np.pad(out, (0, len(sig) - len(out)), mode="edge"), kind


def build_dataset(n_base: int = 300, dup_ratio: float = 0.30, seed: int = 20261124):
    """造 n_base 条原始采集，其中 dup_ratio 比例生成一个"增强副本"。

    副本共享同一 group_id（同一条采集记录）与 session_id（同一采集场次），
    但 sample_id 不同 —— 这正是靠 ID 查不出来的那种重复。
    """
    rng = random.Random(seed)
    npr = np.random.default_rng(seed)
    weights = [0.25, 0.35, 0.25, 0.15]

    X, y, subj, sess, sid, grp, aug, kinds = [], [], [], [], [], [], [], []
    for i in range(n_base):
        label = rng.choices(LABELS, weights=weights)[0]
        s = f"S{rng.randint(1, 40):03d}"
        sig = _wave(label, npr)

        g = f"G{i:04d}"
        X.append(sig); y.append(label); subj.append(s)
        sess.append(f"{s}-{i:04d}"); sid.append(f"SM{i:04d}"); grp.append(g)
        aug.append(False); kinds.append("original")

        if rng.random() < dup_ratio:
            aug_sig, kind = augment(sig, rng, npr)
            X.append(aug_sig); y.append(label); subj.append(s)
            sess.append(f"{s}-{i:04d}"); sid.append(f"SM{i:04d}A"); grp.append(g)
            aug.append(True); kinds.append(kind)

    return {
        "X": np.array(X), "y": y, "subj": subj, "sess": sess,
        "sid": sid, "grp": grp, "aug": aug, "kind": kinds,
    }


# ---------------- 四种指纹 ----------------
def exact_hash(x: np.ndarray) -> str:
    """对原始浮点字节做哈希：任何 1e-16 的差异都会变。"""
    return hashlib.sha256(np.asarray(x, dtype=np.float64).tobytes()).hexdigest()


def downsample(X: np.ndarray, bins: int = BINS) -> np.ndarray:
    """分块求均值：500 点压到 bins 维，抹掉高频抖动、保留动作轮廓。"""
    parts = np.array_split(np.arange(X.shape[1]), bins)
    return np.stack([X[:, p].mean(axis=1) for p in parts], axis=1)


def quant_keys(F: np.ndarray, q: float = Q) -> list[str]:
    """量化指纹：每条样本减去自身均值、除以自身标准差，再量化成整数格子。

    逐样本归一化的作用：抵消整体幅度缩放（戴松/戴紧、设备增益不同），
    这一步能让"幅度 ×1.10"这类副本天然被判为相同。
    """
    keys = []
    for f in F:
        z = (f - f.mean()) / (f.std() + 1e-9)
        cells = np.round(z / q).astype(np.int16)
        keys.append(hashlib.sha256(cells.tobytes()).hexdigest())
    return keys


def bit_signature(F: np.ndarray) -> np.ndarray:
    """二值签名：每一维只保留"高于本样本中位数吗"，得到 bins 个 bit。

    好处：把绝对数值变成相对形状，天然抗幅度差异；
    代价：只留 1 bit，量化误差变成位翻转，必须靠汉明距离容错。
    """
    return (F > np.median(F, axis=1, keepdims=True)).astype(np.uint8)


def logical_key(ds: dict, i: int) -> str:
    """有元数据时的做法：受试者 + 采集场次。副本派生时这两个字段会被保留。"""
    return f"{ds['subj'][i]}|{ds['sess'][i]}"


# ---------------- 评估 ----------------
def measure_keys(keys: list[str], groups: list[str], tr: np.ndarray, te: np.ndarray) -> dict:
    """精确匹配：指纹必须完全相同。"""
    index = defaultdict(set)
    for i in tr:
        index[keys[i]].add(groups[i])
    train_groups = {groups[i] for i in tr}

    tp = fp = fn = tn = 0
    for i in te:
        cand = index.get(keys[i])
        if cand:
            if groups[i] in cand:
                tp += 1
            else:
                fp += 1
        elif groups[i] in train_groups:
            fn += 1
        else:
            tn += 1
    return _stats(tp, fp, fn, tn)


def measure_hamming(sig: np.ndarray, groups: list[str], kinds: list[str],
                    tr: np.ndarray, te: np.ndarray, k: int) -> dict:
    """容忍差异：汉明距离 ≤ k 即视为候选，再看候选里有没有同源样本。"""
    d = np.bitwise_xor(sig[te][:, None, :], sig[tr][None, :, :]).sum(axis=2)
    train_groups = {groups[i] for i in tr}

    tp = fp = fn = tn = 0
    hit_kind: Counter = Counter()
    for r, i in enumerate(te):
        cand = {groups[tr[j]] for j in np.where(d[r] <= k)[0]}
        if cand:
            if groups[i] in cand:
                tp += 1
                hit_kind[kinds[i]] += 1
            else:
                fp += 1
        elif groups[i] in train_groups:
            fn += 1
        else:
            tn += 1
    out = _stats(tp, fp, fn, tn)
    out["hit_kind"] = dict(hit_kind)
    return out


def _stats(tp: int, fp: int, fn: int, tn: int) -> dict:
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="内容级重复检测")
    parser.add_argument("--out", default=str(REPORT))
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()

    ds = build_dataset(seed=args.seed)
    X, groups, kinds = ds["X"], ds["grp"], ds["kind"]

    idx = list(range(len(X)))
    random.Random(42).shuffle(idx)
    cut = int(len(idx) * 0.7)
    tr, te = np.array(idx[:cut]), np.array(idx[cut:])

    F = downsample(X)
    sig = bit_signature(F)

    exact = measure_keys([exact_hash(x) for x in X], groups, tr, te)
    quant = measure_keys(quant_keys(F), groups, tr, te)

    grid = [{"k": k, **measure_hamming(sig, groups, kinds, tr, te, k)} for k in K_GRID]
    ok = [g for g in grid if g["fpr"] <= TARGET_FPR]
    best = max(ok, key=lambda g: g["recall"]) if ok else min(grid, key=lambda g: g["fpr"])

    # 签名长度的影响（汉明=0 固定，只看维度）
    len_grid = []
    for bins in (64, 128, 256):
        s_b = bit_signature(downsample(X, bins))
        r = measure_hamming(s_b, groups, kinds, tr, te, 0)
        len_grid.append({"bins": bins, "recall": r["recall"], "fpr": r["fpr"]})

    keyed = measure_keys([logical_key(ds, i) for i in range(len(X))], groups, tr, te)

    n_aug = sum(ds["aug"])
    test_dup = [i for i in te if groups[i] in {groups[j] for j in tr}]
    dup_by_kind = Counter(kinds[i] for i in test_dup)
    hit_by_kind = best["hit_kind"]

    lines = [
        "# 内容级重复检测报告",
        "",
        "## 一、结论",
        "",
        f"- 样本 **{len(X)}** 条（原始 {len(X) - n_aug} 条 + 增强副本 {n_aug} 条），"
        f"训练 {len(tr)} / 测试 {len(te)}",
        f"- 测试集里跨集同源样本 **{len(test_dup)}** 条 —— 这就是要抓的目标",
        "",
        "| 检测方式 | 检出率 | 误报率 | 前提 |",
        "|---|---|---|---|",
        f"| 精确哈希（原始浮点字节） | **{exact['recall']:.0%}** | {exact['fpr']:.1%} | 无需元数据，但几乎抓不到 |",
        f"| 量化指纹（Q={Q} 精确匹配） | **{quant['recall']:.0%}** | {quant['fpr']:.1%} | 无需元数据 |",
        f"| 二值签名 + 汉明距离 ≤ {best['k']} | **{best['recall']:.0%}** | {best['fpr']:.1%} | 无需元数据，**推荐** |",
        f"| 逻辑键（受试者+场次） | **{keyed['recall']:.0%}** | {keyed['fpr']:.1%} | **需要采集元数据** |",
        "",
        "> 结论不是「二值签名好用」，而是：**有元数据就别用哈希**。",
        "> 哈希是「只剩信号、什么都不知道」时的兜底手段 —— "
        f"在误报 ≤{TARGET_FPR:.0%} 的约束下，最好的一档也只有 {best['recall']:.0%}。",
        "",
        "## 二、为什么精确哈希没用",
        "",
        "同一段信号加了 5% 噪声，浮点数值全变了 —— sha256 会把它们判成两条毫不相关的数据。",
        f"本数据里跨集同源样本 {len(test_dup)} 条，精确哈希只检出 **{exact['tp']}** 条，"
        f"漏检 **{exact['fn']}** 条。",
        "",
        "**这是哈希的固有性质**：它判断「完全相同」，不判断「同源」。",
        "泄漏检测要找的是后者，所以必须先把数据粗化，让微小差异落进同一个格子。",
        "",
        "## 三、三级容错：为什么最后选了汉明距离",
        "",
        "| 做法 | 检出率 | 误报率 | 为什么是这样的结果 |",
        "|---|---|---|---|",
        f"| 量化后精确匹配 | {quant['recall']:.0%} | {quant['fpr']:.1%} | 量化抹平了噪声，"
        "但只要有一维跨格就整体失配 —— 128 维里至少有 1 维跨格几乎必然发生 |",
        f"| 二值化 + 汉明=0 | {grid[0]['recall']:.0%} | {grid[0]['fpr']:.1%} | 二值化把数值差异变成位翻转，"
        "抗幅度、抗噪声都更好，但仍然要求每一 bit 都对 |",
        f"| 二值化 + 汉明≤{best['k']} | {best['recall']:.0%} | {best['fpr']:.1%} | "
        "允许少量位翻转，正好匹配「副本 + 噪声」的差异形态 |",
        "",
        "### 汉明阈值扫描（本数据实测）",
        "",
        "| 汉明距离 ≤ | 检出率 | 误报率 | TP | FP | FN |",
        "|---|---|---|---|---|---|",
    ]
    for g in grid:
        mark = " ← 选中" if g["k"] == best["k"] else ""
        lines.append(
            f"| {g['k']} | {g['recall']:.0%} | {g['fpr']:.1%} | {g['tp']} | {g['fp']} | {g['fn']}{mark} |"
        )

    lines += [
        "",
        f"> 选阈值的规则：**误报率 ≤ {TARGET_FPR:.0%} 的前提下，取检出率最高的一档**。",
        "> 反着来（先定检出率、再容忍误报）会得到一堆「看起来抓了很多」的假重复，",
        "> 最后没人敢用这个检查 —— **误报比漏报更容易毁掉一个检测工具**。",
        "",
        "## 四、按增强类型拆解（这张表最该看）",
        "",
        f"在汉明 ≤ {best['k']} 这一档下，各类型副本的检出情况：",
        "",
        "| 副本类型 | 测试集中的数量 | 检出 | 检出率 |",
        "|---|---|---|---|",
    ]
    for kind, label in KIND_CN.items():
        n = dup_by_kind.get(kind, 0)
        h = hit_by_kind.get(kind, 0)
        rate = f"{h / n:.0%}" if n else "—"
        lines.append(f"| {label} | {n} | {h} | **{rate}** |")

    lines += [
        "",
        "> 注：「原件」那一行的检出率 = 训练集里的副本能否被认出，所以它是三种增强的加权平均。",
        "> 之所以不到六成，是因为其中一部分副本本身就是重采样类型 —— 原件再好认，"
        "对侧那份认不出来也没用。",
        "",
        "**这张表说明了内容哈希的能力边界：**",
        "",
        "- **幅度缩放检出最好**：逐样本归一化直接抵消了增益差异，所以几乎必然命中；",
        "- **加噪类次之**：噪声经分块平均后残差变小，靠汉明容错吃掉；",
        "- **重采样类基本抓不到**：5% 的时轴压缩会把波峰位置整体平移，"
        "**轮廓已经变了，不再属于「轻微扰动」**。要抓它得用 DTW / 互相关做对齐后再比，或者直接查元数据。",
        "",
        "> 所以「对信号做哈希」这句话本身要加限定：**它抓的是扰动型副本，不是变换型副本。**",
        "> 面试时能说出这个边界，比说「我会做感知哈希」值钱。",
        "",
        "## 五、一个反直觉的发现：签名不是越长越好",
        "",
        "| 签名维度 | 汉明=0 检出率 | 误报率 | 说明 |",
        "|---|---|---|---|",
    ]
    notes = {
        64: "分块粗，噪声被平均掉，但区分度低",
        128: "本报告采用的维度",
        256: "**越细越脆**：每块样本点少，噪声残留大，位翻转更多",
    }
    for row in len_grid:
        lines.append(
            f"| {row['bins']} bit | {row['recall']:.0%} | {row['fpr']:.1%} | {notes[row['bins']]} |"
        )

    lines += [
        "",
        "**维度增加带来的是分辨率，也带来噪声敏感性。** 所以调参方向不是「越大越好」，",
        "而是「在误报可接受的前提下，找检出率最高的一档」—— 和阈值选择是同一个逻辑。",
        "",
        "## 六、真正该改的是流程顺序",
        "",
        "| 做法 | 跨集副本数 | 说明 |",
        "|---|---|---|",
        "| 先增强 → 再随机划分 | 见上表 | **错误顺序**：副本与原件分居两侧 |",
        "| 先划分 → 再增强 | 0 | **正确顺序**：增强只发生在各自一侧 |",
        "",
        "> 这条比任何哈希都有效：**只要先划分再增强，内容级泄漏根本不会产生。**",
        "> 检测是事后补救，流程顺序才是根因。",
        "",
        "## 七、局限（要主动说）",
        "",
        "1. 副本由脚本生成，**真实增强副本的差异形态更复杂**（裁剪、拼接、混合），"
        "检出率会低于这里的数字；",
        "2. z-score / 二值化都依赖「本样本自身」的统计量，**同一段信号在不同数据集里签名不同**，"
        "所以签名只在同一次分析内可比，不能跨批次存档比对；",
        "3. 汉明距离是 O(n²) 全量比较，数据量到十万级要换 LSH 分带索引；",
        "4. 本数据只有 4 类动作、40 位受试者，类内多样性低于真实场景，"
        "**误报率在真实数据上会更高**；",
        "5. 静止类信号幅度极低，二值化后接近随机，这类样本的检出率应单独统计。",
        "",
        "## 八、复现",
        "",
        "```bash",
        "python ml_ftest/hash_dup.py",
        "```",
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text(
        json.dumps(
            {
                "samples": len(X),
                "augmented": n_aug,
                "targets": len(test_dup),
                "exact": exact,
                "quantized": quant,
                "best_k": best["k"],
                "best": best,
                "grid": grid,
                "by_kind": {k: {"n": dup_by_kind.get(k, 0), "hit": hit_by_kind.get(k, 0)} for k in KIND_CN},
                "logical_key": keyed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"样本 {len(X)} 条（含副本 {n_aug}），跨集同源 {len(test_dup)} 条")
    print(f"  精确哈希        检出 {exact['recall']:.0%}  误报 {exact['fpr']:.1%}")
    print(f"  量化指纹        检出 {quant['recall']:.0%}  误报 {quant['fpr']:.1%}")
    for g in grid:
        print(f"  汉明≤{g['k']:<2}        检出 {g['recall']:.0%}  误报 {g['fpr']:.1%}")
    print(f"  逻辑键          检出 {keyed['recall']:.0%}  误报 {keyed['fpr']:.1%}")
    print(f"  选中 汉明≤{best['k']}；报告：{out}")


if __name__ == "__main__":
    main()
