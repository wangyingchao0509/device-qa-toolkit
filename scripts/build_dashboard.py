#!/usr/bin/env python3
"""把 reports/ 里的结果汇总成一个自包含的本地看板（单文件 HTML）。

为什么做成单文件：不依赖服务器、不依赖网络，双击就能看，发给人也能看。
图片全部 base64 内嵌，JSON 数据直接嵌进脚本，没有跨域问题。

用法：
    python scripts/build_dashboard.py
    open reports/dashboard.html
"""

from __future__ import annotations

import base64
import html
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RAW = REPORTS / "raw"


def load_json(name: str) -> dict | None:
    p = RAW / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def img_b64(name: str) -> str | None:
    p = REPORTS / name
    if not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode()


def parse_junit() -> dict | None:
    p = RAW / "junit.xml"
    if not p.exists():
        return None
    root = ET.parse(p).getroot()
    suites = root.findall(".//testsuite") or [root]

    total = fails = errors = skipped = 0
    by_module: dict[str, dict[str, int]] = {}
    failed_cases: list[str] = []

    for tc in root.iter("testcase"):
        total += 1
        cls = (tc.get("classname") or "").replace(".", "/")
        mod = cls.split("/")[1] if cls.count("/") >= 1 else cls
        key = f"{cls.rsplit('/', 1)[0]}" if "/" in cls else "其他"
        bucket = by_module.setdefault(key, {"total": 0, "failed": 0, "skipped": 0})
        bucket["total"] += 1
        if tc.find("failure") is not None:
            fails += 1
            bucket["failed"] += 1
            failed_cases.append(tc.get("name", ""))
        elif tc.find("error") is not None:
            errors += 1
            bucket["failed"] += 1
        elif tc.find("skipped") is not None:
            skipped += 1
            bucket["skipped"] += 1

    return {
        "total": total,
        "failed": fails + errors,
        "skipped": skipped,
        "passed": total - fails - errors - skipped,
        "by_module": dict(sorted(by_module.items())),
        "failed_cases": failed_cases,
    }


def analysis_summary() -> list[str]:
    """从分析报告里摘结论段（前几条要点）。"""
    p = REPORTS / "report.md"
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    out, capture = [], False
    for ln in lines:
        if ln.startswith("## 一、结论"):
            capture = True
            continue
        if capture:
            if ln.startswith("## "):
                break
            if ln.strip().startswith("- "):
                out.append(ln.strip()[2:])
    return out


def kw(text: str) -> str:
    """把 **xxx** 转成粗体，其余转义。"""
    parts = text.split("**")
    out = []
    for i, seg in enumerate(parts):
        esc = html.escape(seg)
        out.append(f"<b>{esc}</b>" if i % 2 == 1 else esc)
    return "".join(out)


def badge(passed: bool, text: str = "") -> str:
    cls = "ok" if passed else "bad"
    return f'<span class="badge {cls}">{text or ("通过" if passed else "未通过")}</span>'


def section(title: str, sub: str, body: str) -> str:
    return f"""
    <section class="card">
      <div class="card-head">
        <h2>{title}</h2><span class="sub">{sub}</span>
      </div>
      {body}
    </section>"""


def build() -> str:
    junit = parse_junit()
    perf = load_json("perf_summary.json")
    llm = load_json("llm_eval_run.json")
    judge = load_json("deepeval_result.json")
    concl = analysis_summary()

    # ---------- KPI ----------
    kpis = []
    if junit:
        kpis.append(("自动化用例", f"{junit['passed']}/{junit['total']}",
                     f"失败 {junit['failed']} · 跳过 {junit['skipped']}", junit["failed"] == 0))
    if perf:
        kpis.append(("峰值吞吐", f"{perf['tps']} TPS",
                     f"P95 {perf['p95_ms']} ms · 错误率 {perf['error_rate_pct']}%", perf["error_rate_pct"] < 0.1))
    if llm:
        s = llm["summary"]
        kpis.append(("LLM 通过率", f"{s['pass_rate'] * 100:.0f}%",
                     f"{s['passed']}/{s['total_cases']} · 判官 Kappa 见下", s["pass_rate"] >= 0.85))
    if judge:
        kpis.append(("判官通过率", f"{judge['pass_rate'] * 100:.0f}%",
                     f"{judge['passed']}/{judge['cases']} · 正确性 {judge['avg_correctness']}", True))

    kpi_html = "".join(
        f'<div class="kpi"><div class="kpi-label">{k}</div>'
        f'<div class="kpi-value {"ok-text" if ok else "bad-text"}">{v}</div>'
        f'<div class="kpi-note">{html.escape(n)}</div></div>'
        for k, v, n, ok in kpis
    )

    # ---------- 接口测试 ----------
    api_body = "<p class='muted'>未找到 reports/raw/junit.xml，先跑 <code>make dashboard</code>。</p>"
    if junit:
        rows = "".join(
            f"<tr><td>{html.escape(mod)}</td><td>{v['total']}</td>"
            f"<td class='{'ok-text' if v['failed'] == 0 else 'bad-text'}'>{v['failed']}</td>"
            f"<td>{v['skipped']}</td></tr>"
            for mod, v in junit["by_module"].items()
        )
        api_body = f"""
        <table><thead><tr><th>模块</th><th>用例数</th><th>失败</th><th>跳过</th></tr></thead>
        <tbody>{rows}</tbody></table>
        <p class="muted">共 {junit['total']} 条，通过 {junit['passed']}，失败 {junit['failed']}，跳过 {junit['skipped']}。
        「跳过」是 DeepEval 可选集成：未安装或未配置 API Key 时自动跳过，保证克隆即可运行。</p>
        """

    # ---------- 性能 ----------
    perf_body = "<p class='muted'>未找到 perf_summary.json，先跑 <code>make perf</code>。</p>"
    if perf:
        sc_rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{v['count']}</td><td>{v['avg_ms']} ms</td>"
            f"<td>{v['p95_ms']} ms</td><td>{v['error_rate_pct']}%</td></tr>"
            for k, v in perf.get("scenarios", {}).items()
        )
        img = img_b64("perf_throughput.png")
        perf_body = f"""
        <div class="grid2">
          <div>
            <table><tbody>
              <tr><th>总请求数</th><td>{perf['requests']}</td></tr>
              <tr><th>平均 TPS</th><td>{perf['tps']}</td></tr>
              <tr><th>平均 / P95 / P99</th><td>{perf['avg_ms']} / {perf['p95_ms']} / {perf['p99_ms']} ms</td></tr>
              <tr><th>错误率</th><td>{perf['error_rate_pct']}%</td></tr>
              <tr><th>持续时间</th><td>{perf['duration_s']} s</td></tr>
            </tbody></table>
          </div>
          <div>{f'<img src="data:image/png;base64,{img}" alt="perf">' if img else ''}</div>
        </div>
        <table><thead><tr><th>场景</th><th>请求数</th><th>平均 RT</th><th>P95</th><th>错误率</th></tr></thead>
        <tbody>{sc_rows}</tbody></table>
        """

    # ---------- 数据分析 ----------
    ana_body = "<p class='muted'>未找到 reports/report.md。</p>"
    if concl:
        imgs = [img_b64(n) for n in ("capacity_fade.png", "temp_distribution.png", "pass_overview.png")]
        gal = "".join(
            f'<img src="data:image/png;base64,{b}" alt="chart">' for b in imgs if b
        )
        ana_body = (
            "<ul>" + "".join(f"<li>{kw(t)}</li>" for t in concl) + "</ul>"
            + f'<div class="gallery">{gal}</div>'
        )

    # ---------- LLM ----------
    llm_body = "<p class='muted'>未找到 llm_eval_run.json，先跑 <code>make llm-evals</code>。</p>"
    if llm:
        s = llm["summary"]
        cal = llm.get("calibration") or {}
        rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{v['passed']}/{v['total']}</td>"
            f"<td>{v['pass_rate'] * 100:.0f}%</td>"
            f"<td>{badge(v['pass_rate'] == 1, 'PASS' if v['pass_rate'] == 1 else 'FAIL')}</td></tr>"
            for k, v in s["by_intent"].items()
        )
        fail_rows = "".join(
            f"<tr><td><code>{html.escape(f['id'])}</code></td><td>{html.escape(f['input'][:16])}</td>"
            f"<td class='small'>{html.escape(f['answer'][:36])}</td>"
            f"<td class='small'>{html.escape('；'.join(f['reasons']))}</td></tr>"
            for f in s["failures"]
        )
        cmp_html = ""
        if judge:
            cmp_html = f"""
            <h3>规则判分 vs 判官判分</h3>
            <table>
              <thead><tr><th>判分方式</th><th>通过</th><th>通过率</th><th>说明</th></tr></thead>
              <tbody>
                <tr><td>规则判分（本仓库）</td><td>{s['passed']}/{s['total_cases']}</td>
                    <td>{s['pass_rate'] * 100:.0f}%</td><td class="small">确定性断言，零成本</td></tr>
                <tr><td>DeepEval + DeepSeek 判官</td><td>{judge['passed']}/{judge['cases']}</td>
                    <td>{judge['pass_rate'] * 100:.0f}%</td>
                    <td class="small">正确性 {judge['avg_correctness']} / 相关性 {judge['avg_relevancy']}</td></tr>
              </tbody>
            </table>
            <p class="muted">两次都有差异的一条是 <code>BOUND-003</code>：规则判通过，判官给 0 分，
            而它给出的理由恰好描述的是正确行为——分数与理由自相矛盾。这就是"确定性断言必须用代码"的实证。</p>
            """
        llm_body = f"""
        <div class="meta">
          模型 <b>{html.escape(llm.get('model',''))}</b> ·
          提示词 <code>{html.escape(llm.get('prompt_version',''))}</code> ·
          数据集 <code>{html.escape(llm.get('dataset_version',''))}</code> ·
          temperature={llm.get('temperature')} · seed={llm.get('seed')}
        </div>
        <div class="grid3">
          <div class="mini"><span>幻觉</span><b class="bad-text">{s['hallucination_count']}</b></div>
          <div class="mini"><span>注入泄露</span><b class="bad-text">{s['injection_leak_count']}</b></div>
          <div class="mini"><span>多轮一致性</span><b>{s['consistency'] * 100:.0f}%</b></div>
          <div class="mini"><span>平均延迟</span><b>{s['avg_latency_ms']:.0f} ms</b></div>
          <div class="mini"><span>总 token</span><b>{s['total_tokens']}</b></div>
          <div class="mini"><span>估算成本</span><b>${s['estimated_cost_usd']}</b></div>
        </div>
        <h3>分类通过率</h3>
        <table><thead><tr><th>意图分类</th><th>通过/总数</th><th>通过率</th><th>结论</th></tr></thead>
        <tbody>{rows}</tbody></table>
        <h3>未通过明细</h3>
        <table><thead><tr><th>用例</th><th>输入</th><th>实际回答</th><th>失败原因</th></tr></thead>
        <tbody>{fail_rows}</tbody></table>
        {cmp_html}
        """
        if cal:
            llm_body += (
                f"<h3>判官校准</h3><p>人工标注 {cal.get('n')} 条，观察一致率 "
                f"{cal.get('agreement', 0) * 100:.0f}%，<b>Cohen's Kappa = {cal.get('kappa')}</b>"
                f"（{'可用' if cal.get('usable') else '不可用'}，阈值 0.6）。</p>"
            )

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Device QA Toolkit · 测试看板</title>
<style>
  :root {{
    --bg:#f5f6f8; --card:#fff; --ink:#1f2328; --muted:#6b7280;
    --line:#e3e6ea; --brand:#1f3864; --ok:#0f7b3f; --bad:#c0392b;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; background:var(--bg); color:var(--ink);
    font:14px/1.65 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
  }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:28px 20px 60px; }}
  header {{ margin-bottom:20px; }}
  header h1 {{ margin:0 0 4px; font-size:22px; letter-spacing:.5px; color:var(--brand); }}
  header .meta {{ color:var(--muted); font-size:12.5px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin:18px 0; }}
  .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
  .kpi-label {{ font-size:12px; color:var(--muted); }}
  .kpi-value {{ font-size:24px; font-weight:600; margin:2px 0; }}
  .kpi-note {{ font-size:11.5px; color:var(--muted); }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px 20px; margin-bottom:16px; }}
  .card-head {{ display:flex; align-items:baseline; gap:10px; border-bottom:1px solid var(--line); padding-bottom:8px; margin-bottom:12px; }}
  .card-head h2 {{ margin:0; font-size:15px; color:var(--brand); }}
  .sub {{ font-size:11.5px; color:var(--muted); }}
  h3 {{ font-size:13px; margin:16px 0 8px; color:#2f5496; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; margin:6px 0 10px; }}
  th,td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:left; vertical-align:top; }}
  th {{ background:#fafbfc; font-weight:600; color:#374151; font-size:12px; }}
  code {{ background:#f3f4f6; padding:1px 5px; border-radius:4px; font-size:12px; }}
  .ok-text {{ color:var(--ok); }} .bad-text {{ color:var(--bad); }}
  .badge {{ display:inline-block; padding:1px 8px; border-radius:99px; font-size:11px; }}
  .badge.ok {{ background:#e6f4ec; color:var(--ok); }}
  .badge.bad {{ background:#fdecea; color:var(--bad); }}
  .muted {{ color:var(--muted); font-size:12.5px; }}
  .small {{ font-size:12px; color:#4b5563; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; align-items:start; }}
  .grid3 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; margin:10px 0; }}
  .mini {{ background:#fafbfc; border:1px solid var(--line); border-radius:8px; padding:10px 12px; }}
  .mini span {{ display:block; font-size:11.5px; color:var(--muted); }}
  .mini b {{ font-size:16px; }}
  .gallery {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:10px; margin-top:10px; }}
  img {{ width:100%; border:1px solid var(--line); border-radius:8px; }}
  .meta {{ font-size:12.5px; color:var(--muted); margin-bottom:8px; }}
  footer {{ text-align:center; color:var(--muted); font-size:11.5px; margin-top:24px; }}
  @media (max-width:720px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Device QA Toolkit · 测试看板</h1>
    <div class="meta">生成时间 {ts}　·　
      <a href="https://github.com/wangyingchao0509/device-qa-toolkit">GitHub 仓库</a>
    </div>
  </header>

  <div class="kpis">{kpi_html}</div>

  {section("接口自动化", "pytest · 边界值 / 幂等去重 / 并发一致性 / 契约校验", api_body)}
  {section("负载测试", "场景配比 上报 55% / 查询 30% / 汇总 15%", perf_body)}
  {section("试验数据分析", "pandas 清洗 → 判定口径 → matplotlib 出图", ana_body)}
  {section("大模型评测", "规则判分 + 可选判官，三件套版本绑定", llm_body)}

  <footer>本页为本地静态看板，数据来自 reports/ 下的产物，不联网、不含任何外发请求。</footer>
</div>
</body>
</html>"""


def main() -> None:
    html_text = build()
    out = REPORTS / "dashboard.html"
    out.write_text(html_text, encoding="utf-8")
    size = os.path.getsize(out) / 1024
    print(f"看板已生成：{out}（{size:.0f} KB）")
    print("打开方式：open reports/dashboard.html")


if __name__ == "__main__":
    main()
