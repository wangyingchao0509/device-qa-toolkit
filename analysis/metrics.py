"""试验数据分析：把原始采集数据变成可判读的结论。

这是本仓库和普通「接口自动化 demo」的区别所在：
硬件/穿戴测试每天产出的是大量原始数值（容量、温升、心率曲线），
测试人员真正的价值是把这些数据整理成"能不能放行"的判断依据。
"""

from __future__ import annotations

import pandas as pd

# 判定口径（真实项目中来自规格书或企业标准，这里作为示例常量集中管理）
RATED_CAPACITY_MAH = 500
CAPACITY_PASS_RATIO = 0.80      # 循环后容量保持率 ≥ 80%
TEMP_PASS_LIMIT_C = 45.0        # 表面温升上限
SPEC_TEMP_RANGE = (-20.0, 60.0)


def load_readings(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """拆分正常数据与异常数据，返回 (clean_df, rejected_df)。

    清洗规则：
    1. 关键字段缺失 → 剔除（记录数要能对上）
    2. 超出传感器量程 → 剔除（不是产品问题，是采集问题）
    3. 同一设备同一 cycle 重复 → 保留最后一条（设备重传）
    """
    required = ["sn", "cycle", "capacity_mah", "temp_c"]
    missing = df[df[required].isna().any(axis=1)]

    out_of_spec = df[
        (~df.index.isin(missing.index))
        & ((df["temp_c"] < SPEC_TEMP_RANGE[0]) | (df["temp_c"] > SPEC_TEMP_RANGE[1]))
    ]

    valid = df.drop(index=missing.index.union(out_of_spec.index))
    valid = valid.sort_values("ts").drop_duplicates(subset=["sn", "cycle"], keep="last")

    rejected = pd.concat([missing.assign(reason="missing_field"), out_of_spec.assign(reason="out_of_spec")])
    return valid.reset_index(drop=True), rejected.reset_index(drop=True)


def per_device_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """按设备分组统计：容量保持率、P95 温升、判定结果。"""
    rows = []
    for sn, g in df.groupby("sn"):
        g = g.sort_values("cycle")
        initial = g["capacity_mah"].iloc[0]
        last = g["capacity_mah"].iloc[-1]
        ratio = last / initial if initial else 0.0
        p95_temp = g["temp_c"].quantile(0.95)
        rows.append(
            {
                "sn": sn,
                "cycles": int(g["cycle"].max()),
                "initial_mah": round(initial, 1),
                "last_mah": round(last, 1),
                "retention_pct": round(ratio * 100, 2),
                "p95_temp_c": round(float(p95_temp), 2),
                "max_temp_c": round(float(g["temp_c"].max()), 2),
                "capacity_pass": ratio >= CAPACITY_PASS_RATIO,
                "temp_pass": p95_temp <= TEMP_PASS_LIMIT_C,
            }
        )
    out = pd.DataFrame(rows)
    out["overall_pass"] = out["capacity_pass"] & out["temp_pass"]
    return out.sort_values("sn").reset_index(drop=True)


def overall_summary(metrics: pd.DataFrame) -> dict:
    total = len(metrics)
    passed = int(metrics["overall_pass"].sum())
    return {
        "devices": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate_pct": round(passed / total * 100, 1) if total else 0.0,
        "worst_retention_pct": round(float(metrics["retention_pct"].min()), 2) if total else None,
        "highest_p95_temp_c": round(float(metrics["p95_temp_c"].max()), 2) if total else None,
    }
