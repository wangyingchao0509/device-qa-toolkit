"""ML 功能测试工具的测试。

这三类工具的测试重点不是"模型准不准"，而是：
① 泄漏检测的判定逻辑正确（分组划分不泄漏、ID 重复能识别）；
② 蜕变关系确实只改了它该改的特征；
③ 分类器可复现（否则不变率没有意义）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_ftest.leakage import id_duplicates, split_by_subject, split_random
from ml_ftest.metamorphic import (
    FEATURES,
    NearestCentroid,
    gen_features,
    mr_drop_feature,
    mr_freq_shift,
    mr_scale,
    mr_zero,
)

ROOT_CSV = "data/ml/slice_sample.csv"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(ROOT_CSV)


@pytest.mark.ml
def test_group_split_has_no_subject_leakage(df):
    train, test = split_by_subject(df)
    assert not (set(train.subject_id) & set(test.subject_id)), "分组划分不允许受试者交叉"


@pytest.mark.ml
def test_random_split_leaks_subjects(df):
    """随机划分几乎必然泄漏 —— 这是我们要证明的反面案例。"""
    train, test = split_random(df)
    assert len(set(train.subject_id) & set(test.subject_id)) > 0


@pytest.mark.ml
def test_id_duplicates_detects_dup():
    d = pd.DataFrame({"sample_id": ["A", "A", "B"]})
    assert id_duplicates(d) == 1


@pytest.mark.ml
def test_mr_scale_only_touches_amplitude_features():
    X, _, _ = gen_features(n=5)
    Y = mr_scale(X, 1.1)
    for i, name in enumerate(FEATURES):
        if name in ("mean_acc", "std_acc", "spectral_energy"):
            assert np.allclose(Y[:, i], X[:, i] * 1.1), f"{name} 应被缩放"
        else:
            assert np.allclose(Y[:, i], X[:, i]), f"{name} 不应被改动"


@pytest.mark.ml
def test_mr_freq_shift_touches_only_freq():
    X, _, _ = gen_features(n=5)
    Y = mr_freq_shift(X, 1.05)
    idx = FEATURES.index("dominant_freq")
    for i in range(len(FEATURES)):
        if i == idx:
            assert np.allclose(Y[:, i], X[:, i] * 1.05)
        else:
            assert np.allclose(Y[:, i], X[:, i])


@pytest.mark.ml
def test_mr_zero_is_all_zero():
    X, _, _ = gen_features(n=5)
    assert np.allclose(mr_zero(X), 0.0)


@pytest.mark.ml
def test_mr_drop_feature_zeroes_target():
    X, _, _ = gen_features(n=5)
    Y = mr_drop_feature(X, "zcr")
    assert np.allclose(Y[:, FEATURES.index("zcr")], 0.0)


@pytest.mark.ml
def test_classifier_is_deterministic():
    X, y, _ = gen_features(n=200)
    a = NearestCentroid().fit(X[:150], y[:150]).predict(X[150:])
    b = NearestCentroid().fit(X[:150], y[:150]).predict(X[150:])
    assert a == b, "分类器必须可复现，否则不变率没有可比性"


@pytest.mark.ml
def test_zero_input_predicted_as_still():
    """MR-5：全零输入应判「静止」，判成跑步就是没学到物理规律。"""
    X, y, _ = gen_features(n=300)
    model = NearestCentroid().fit(X, y)
    assert model.predict(mr_zero(X[:1]))[0] == "静止"


# ---------------- 内容级重复检测（hash_dup） ----------------
from ml_ftest.hash_dup import (  # noqa: E402
    bit_signature,
    build_dataset,
    downsample,
    exact_hash,
    logical_key,
    measure_hamming,
)


@pytest.mark.ml
def test_exact_hash_breaks_on_tiny_noise():
    """精确哈希对 1e-12 的差异都会失效——这是它的固有性质，不是 bug。"""
    x = np.arange(100.0)
    assert exact_hash(x) != exact_hash(x + 1e-12)


@pytest.mark.ml
def test_bit_signature_survives_amplitude_scale():
    """整体缩放不改变二值签名（比的是逐样本中位数，不是绝对值）。"""
    ds = build_dataset(n_base=20, dup_ratio=0.0, seed=7)
    X = ds["X"][:5]
    assert np.array_equal(
        bit_signature(downsample(X)), bit_signature(downsample(X * 1.10))
    )


@pytest.mark.ml
def test_downsample_shape():
    X = np.random.default_rng(0).normal(size=(3, 500))
    assert downsample(X, 32).shape == (3, 32)


@pytest.mark.ml
def test_logical_key_shared_by_copy():
    """副本派生时保留采集场次，所以逻辑键相同——这是它比哈希更准的原因。"""
    ds = build_dataset(n_base=10, dup_ratio=1.0, seed=7)
    assert logical_key(ds, 0) == logical_key(ds, 1)


@pytest.mark.ml
def test_hamming_hits_same_source():
    sig = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 1]], dtype=np.uint8)
    groups, kinds = ["A", "A", "B"], ["original", "copy", "original"]
    r = measure_hamming(sig, groups, kinds, np.array([0, 2]), np.array([1]), k=1)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 0)


@pytest.mark.ml
def test_hamming_reports_false_positive():
    """近但不同源 → 必须计入误报，不能算检出。"""
    sig = np.array([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.uint8)
    groups, kinds = ["A", "B"], ["original", "copy"]
    r = measure_hamming(sig, groups, kinds, np.array([0]), np.array([1]), k=1)
    assert (r["tp"], r["fp"]) == (0, 1)
