"""A local-only demonstration API; no external accounts or production writes."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from urllib.parse import urlsplit

from .dataset import SUITES, SCHEMA_VERSION, semantic_group, task_for
from .domains import DOMAINS, candidate_worlds
from .evaluation import rollout
from .io import digest
from .policies import StaticPolicy
from .report import STYLE


def demo_run(payload):
    domain = payload.get("domain", "travel")
    suite = payload.get("suite", "mixed")
    policy = payload.get("policy", "static")
    seed = payload.get("seed", 0)
    variant = payload.get("variant", 0)
    if domain not in DOMAINS or suite not in SUITES or policy not in ("static", "oracle", "refresh", "identify", "legacy", "schema"):
        raise ValueError("Unsupported selection")
    if type(seed) is not int or not 0 <= seed <= 10000 or type(variant) is not int:
        raise ValueError("Invalid seed or variant")
    task = task_for(seed, domain)
    worlds = candidate_worlds(domain, suite)
    if not 0 <= variant < len(worlds):
        raise ValueError("Variant is outside the selected support")
    row = dict(schema_version=SCHEMA_VERSION, episode_id=digest([domain, suite, seed, variant])[:24],
               group_id=semantic_group(task), split="demo", evaluation_category="demo",
               domain=domain, suite=suite, task=task.to_dict(), world=worlds[variant].to_dict(),
               support=[w.to_dict() for w in worlds])
    return rollout(row, None if policy == "oracle" else StaticPolicy(policy), policy)


PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ToolShift · 工具变化实验台</title><style>""" + STYLE + """</style><main>
<p class="muted">TOOLSHIFT / LOCAL REPLAY</p><h1>工具变化实验台</h1>
<p>选择业务、变化组合和策略。运行会创建独立 SQLite 状态，执行真实工具写入，再检查目标与受保护记录。全部操作发生在本机合成环境。</p>
<div class="card"><label>业务 <select id="domain"><option>travel</option><option>catalog</option><option>calendar</option></select></label>
<label>变化 <select id="suite"><option>mixed</option><option>units</option><option>append_replace</option><option>default_scope</option><option>no_drift</option><option>unidentifiable</option></select></label>
<label>策略 <select id="policy"><option>static</option><option>oracle</option><option>refresh</option><option>identify</option><option>legacy</option><option>schema</option></select></label>
<label>版本 <select id="variant"><option>0</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option><option>6</option><option>7</option></select></label>
<button id="run">运行并验证</button></div><div id="status" class="note">选择 mixed + 版本 7，比较 static 与 legacy。</div>
<section id="trace"></section><details class="card"><summary>完整验证记录</summary><pre id="raw"></pre></details>
<p class="muted">公开候选语义范围是此基准的假设。三个域共享两个操作机制；该演示不证明真实 API 泛化或 LLM 改进。</p></main>
<script>
const $=id=>document.getElementById(id);
$('suite').onchange=()=>{const count=$('suite').value==='mixed'?8:2;const selected=Number($('variant').value);$('variant').replaceChildren();for(let i=0;i<count;i++){const option=document.createElement('option');option.value=String(i);option.textContent=String(i);$('variant').appendChild(option)}$('variant').value=String(Math.min(selected,count-1));};
$('run').onclick=async()=>{const button=$('run');button.disabled=true;$('status').textContent='正在创建数据库与执行…';$('trace').replaceChildren();
try{const response=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({domain:$('domain').value,suite:$('suite').value,policy:$('policy').value,variant:Number($('variant').value),seed:17})});
const data=await response.json();if(!response.ok)throw Error(data.error);
$('status').textContent=`实际成功: ${data.success} · 有证据的成功: ${data.robust_success} · 不变量保持: ${data.invariant_ok} · 查询成本: ${data.query_cost} · 结束: ${data.termination}`;
for(const step of data.trace){const card=document.createElement('div');card.className='card';const pre=document.createElement('pre');pre.textContent=JSON.stringify(step,null,2);card.appendChild(pre);$('trace').appendChild(card)}
$('raw').textContent=JSON.stringify(data,null,2);
}catch(error){$('status').textContent=error.message}finally{button.disabled=false}};
</script></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "ToolShift/0.1"

    def respond(self, code, content, content_type="application/json; charset=utf-8"):
        data = content.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self.respond(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/health":
            self.respond(200, '{"ok":true}')
        else:
            self.respond(404, '{"error":"not found"}')

    def do_POST(self):
        if self.path != "/api/run":
            return self.respond(404, '{"error":"not found"}')
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            return self.respond(403, '{"error":"cross-origin request rejected"}')
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 1 <= size <= 65536:
                raise ValueError("Request body must be 1–65536 bytes")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            self.respond(200, json.dumps(demo_run(payload), ensure_ascii=False, allow_nan=False))
        except (ValueError, TypeError, KeyError) as exc:
            self.respond(400, json.dumps({"error": str(exc)}))


def serve(port=8765):
    with HTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"ToolShift running at http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass

