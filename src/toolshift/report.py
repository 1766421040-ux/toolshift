"""Self-contained HTML reports generated only from measured results."""
from __future__ import annotations

import html
import json
from pathlib import Path


STYLE = """
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#17243a;font:16px/1.65 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:40px 24px}
h1{font-size:38px;letter-spacing:-1px}h2{margin-top:36px}a{color:#235ec4}p{max-width:880px}
.note{background:#e7effc;border-left:4px solid #396ed2;padding:16px}.card{background:white;border:1px solid #dfe5ee;border-radius:12px;padding:22px;margin:18px 0;overflow:auto}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:11px;border-bottom:1px solid #e4e9f0;white-space:nowrap}th{background:#edf2f9}code,pre{font-family:ui-monospace,monospace;font-size:13px}pre{white-space:pre-wrap;overflow-wrap:anywhere}select,button{font:inherit;padding:8px;border:1px solid #bcc9dc;border-radius:6px}button{background:#245cb7;color:white;cursor:pointer}.muted{color:#62728a}.ok{color:#116c4f}.fail{color:#ab2332}
"""


def _number(value, percent=False):
    if value is None:
        return "—"
    return f"{value * 100:.1f}%" if percent else f"{value:.3f}"


def render_report(results: list[dict], output: str | Path, title="ToolShift · 实验报告") -> Path:
    rows = []
    for result in results:
        s = result["summary"]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in [
            result["policy"], s["episodes"], _number(s.get("success_rate"), True),
            _number(s.get("robust_success_rate"), True),
            _number(s.get("feasible_robust_success_rate"), True),
            _number(s.get("mean_query_cost")), _number(s.get("query_cost_per_success")),
            _number(s.get("invariant_violation_rate"), True),
            _number(s.get("abstention_rate"), True),
        ]) + "</tr>")
    details = "".join(
        f'<details class="card"><summary>{html.escape(r["policy"])} · 分组结果与运行环境</summary>'
        f'<pre>{html.escape(json.dumps(r, ensure_ascii=False, indent=2))}</pre></details>'
        for r in results)
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{STYLE}</style><main><p class="muted">ToolShift / 实际运行记录</p><h1>{html.escape(title)}</h1>
<p>工具语义变化后的主动诊断与任务恢复。每个结果来自 SQLite 中的实际写入及独立终态验证。候选变化范围是所有策略共享的公开先验。</p>
<div class="note">“实际成功”可以包含猜中；“稳健成功”还要求每次写入在当时的整个候选范围内有可完成性证据。暂停不计成功；“每次成功的查询成本”包含失败样本已经消耗的查询。没有成功时该指标为空。
有限合成环境结果不等同于真实 API、GUI 或未知世界泛化；三个业务域共享两类操作机制。</div>
<section class="card"><table><thead><tr><th>策略</th><th>任务数</th><th>实际成功</th><th>稳健成功</th><th>可完成子集稳健成功</th><th>平均查询成本</th><th>查询成本/实际成功</th><th>约束违反</th><th>暂停率</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
{details}<p class="muted">完整逐步轨迹、配置和原始汇总位于对应运行目录。没有填入未运行的实验数据。</p></main></html>"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output

