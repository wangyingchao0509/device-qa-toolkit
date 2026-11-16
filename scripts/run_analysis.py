"""一键生成分析报告：python scripts/run_analysis.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.report import run  # noqa: E402

if __name__ == "__main__":
    csv_path = ROOT / "data" / "sample" / "battery_cycles.csv"
    if not csv_path.exists():
        print("未找到样例数据，先运行：python scripts/gen_sample_data.py")
        raise SystemExit(1)

    report_path = run(str(csv_path), out_dir=str(ROOT / "reports"))
    print(f"报告已生成：{report_path}")
    print("图表目录：" + str(ROOT / "reports"))
