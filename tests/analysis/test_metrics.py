"""分析模块的单元测试：数据清洗与判定口径必须可验证。

为什么测试人员也要写这个：
报告里"通过率 60%"这个数字，如果清洗逻辑写错，整份报告就是错的。
所以口径本身也要有测试，而不是靠人工核对。
"""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.metrics import (
    CAPACITY_PASS_RATIO,
    TEMP_PASS_LIMIT_C,
    clean,
    overall_summary,
    per_device_metrics,
)


def _df(rows):
    return pd.DataFrame(rows, columns=["sn", "cycle", "capacity_mah", "temp_c", "ts"])


def test_clean_drops_missing_and_out_of_spec():
    df = _df(
        [
            ["SN1", 1, 500.0, 35.0, 1],
            ["SN1", 2, None, 36.0, 2],      # 缺容量 -> 剔除
            ["SN1", 3, 498.0, 999.0, 3],    # 超量程 -> 剔除
            ["SN1", 4, 497.0, 37.0, 4],
        ]
    )
    kept, rejected = clean(df)
    assert len(kept) == 2
    assert len(rejected) == 2
    assert set(rejected["reason"]) == {"missing_field", "out_of_spec"}


def test_clean_dedup_keeps_latest_by_ts():
    df = _df(
        [
            ["SN1", 1, 500.0, 35.0, 10],
            ["SN1", 1, 490.0, 35.0, 20],   # 同一 cycle 重传，保留时间更晚的
        ]
    )
    kept, _ = clean(df)
    assert len(kept) == 1
    assert kept.iloc[0]["capacity_mah"] == 490.0


def test_retention_and_pass_flag():
    df = _df(
        [
            ["SN1", 1, 500.0, 35.0, 1],
            ["SN1", 500, 410.0, 36.0, 2],   # 82% -> 容量通过
            ["SN2", 1, 500.0, 40.0, 1],
            ["SN2", 500, 380.0, 40.0, 2],   # 76% -> 容量不通过
        ]
    )
    metrics = per_device_metrics(clean(df)[0]).set_index("sn")

    assert metrics.loc["SN1", "retention_pct"] == pytest.approx(82.0, abs=0.1)
    assert metrics.loc["SN1", "capacity_pass"]
    assert not metrics.loc["SN2", "capacity_pass"]
    assert metrics.loc["SN2", "retention_pct"] < CAPACITY_PASS_RATIO * 100


def test_temp_pass_uses_p95_not_max():
    """温升判据用 P95：偶发单点尖峰不应直接判死，但持续偏高必须拦住。"""
    rows = [["SN1", i, 500 - i, 30.0, i] for i in range(1, 21)]
    rows += [["SN1", 21, 470, 46.0, 21]]         # 单个尖峰
    metrics = per_device_metrics(clean(_df(rows))[0]).iloc[0]

    assert metrics["max_temp_c"] == 46.0
    assert metrics["p95_temp_c"] <= TEMP_PASS_LIMIT_C
    assert metrics["temp_pass"] is not False


def test_overall_summary_counts():
    df = _df(
        [
            ["SN1", 1, 500.0, 35.0, 1],
            ["SN1", 500, 450.0, 35.0, 2],   # 通过
            ["SN2", 1, 500.0, 35.0, 1],
            ["SN2", 500, 350.0, 35.0, 2],   # 容量不通过
        ]
    )
    summary = overall_summary(per_device_metrics(clean(df)[0]))
    assert summary["devices"] == 2
    assert summary["passed"] == 1
    assert summary["pass_rate_pct"] == 50.0
