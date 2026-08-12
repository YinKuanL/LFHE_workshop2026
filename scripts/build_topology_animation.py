#!/usr/bin/env python3
"""Build a self-contained interactive topology-evolution viewer.

The viewer reconstructs PAC topology states from graph_initial.edgelist and
pac_edge_deltas.jsonl.  It is safe to run while an experiment is appending to
the JSONL file: an incomplete final line is ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import networkx as nx


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # A running process may be part-way through its last append.
                continue
    return rows


def _edge(edge) -> list[int]:
    a, b = map(int, edge[:2])
    return [a, b] if a <= b else [b, a]


def _read_edges(path: Path) -> list[list[int]]:
    edges: set[tuple[int, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) >= 2:
                edges.add(tuple(_edge(fields)))
    return [list(edge) for edge in sorted(edges)]


def _layout(num_clients: int, edges: list[list[int]]) -> list[list[float]]:
    graph = nx.Graph()
    graph.add_nodes_from(range(num_clients))
    graph.add_edges_from(edges)
    positions = nx.spring_layout(
        graph,
        seed=17,
        iterations=80 if num_clients <= 200 else 45,
        scale=1.0,
    )
    return [[round(float(positions[i][0]), 5), round(float(positions[i][1]), 5)] for i in range(num_clients)]


def collect_runs(outputs: Path, methods: set[str] | None = None) -> list[dict]:
    found: dict[tuple[str, int, int], tuple[float, dict]] = {}
    layout_cache: dict[str, list[list[float]]] = {}
    for config_path in outputs.rglob("config.json"):
        run_dir = config_path.parent
        initial_path = run_dir / "graph_initial.edgelist"
        delta_path = run_dir / "pac_edge_deltas.jsonl"
        if not initial_path.exists() or not delta_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            method = str(config["method"])
            n = int(config["num_clients"])
            seed = int(config["seed"])
        except (KeyError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if methods and method not in methods:
            continue
        initial_edges = _read_edges(initial_path)
        cache_key = hashlib.sha256(
            json.dumps([n, initial_edges], separators=(",", ":")).encode()
        ).hexdigest()
        if cache_key not in layout_cache:
            layout_cache[cache_key] = _layout(n, initial_edges)
        frames = []
        for row in _read_jsonl(delta_path):
            try:
                frames.append(
                    {
                        "r": int(row["round"]),
                        "a": [_edge(edge) for edge in row.get("added", [])],
                        "d": [_edge(edge) for edge in row.get("removed", [])],
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        frames.sort(key=lambda frame: frame["r"])
        record = {
            "m": method,
            "n": n,
            "s": seed,
            "dmax": int(config.get("dmax", config.get("dmax_spec", 0))),
            "path": str(run_dir),
            "initial": initial_edges,
            "positions": layout_cache[cache_key],
            "frames": frames,
            "complete": (run_dir / "SUCCESS").exists(),
        }
        key = (method, n, seed)
        modified = delta_path.stat().st_mtime
        # If an overly broad output root contains duplicates, prefer the run
        # whose delta log was updated most recently.
        if key not in found or modified > found[key][0]:
            found[key] = (modified, record)
    return [
        item[1]
        for item in sorted(
            found.values(), key=lambda item: (item[1]["m"], item[1]["n"], item[1]["s"])
        )
    ]


def render_html(runs: list[dict], title: str) -> str:
    payload = json.dumps(runs, separators=(",", ":"), ensure_ascii=False)
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_title}</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui,sans-serif; }}
body {{ margin: 0; padding: 18px; background: Canvas; color: CanvasText; }}
main {{ max-width: 1180px; margin: auto; }}
h1 {{ font-size: 1.25rem; font-weight: 500; margin: 0 0 14px; }}
.controls {{ display:flex; flex-wrap:wrap; gap:10px 14px; align-items:end; margin-bottom:12px; }}
label {{ display:grid; gap:4px; font-size:.82rem; }}
select,button,input {{ font:inherit; }}
select,button {{ padding:6px 9px; }}
input[type=range] {{ width:min(360px,70vw); }}
.timeline {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin:8px 0; }}
.stage {{ border:1px solid color-mix(in srgb, CanvasText 22%, transparent); min-height:420px; }}
svg {{ width:100%; height:auto; display:block; }}
.base {{ stroke:color-mix(in srgb, CanvasText 28%, transparent); stroke-width:1; }}
.added {{ stroke:#19a55a; stroke-width:2.4; }}
.removed {{ stroke:#d64545; stroke-width:2.4; stroke-dasharray:5 3; }}
.node {{ fill:Canvas; stroke:CanvasText; stroke-width:1.1; }}
.legend,.status {{ display:flex; flex-wrap:wrap; gap:14px; font-size:.85rem; margin:9px 0; }}
.swatch {{ display:inline-block; width:22px; border-top:3px solid; vertical-align:middle; margin-right:5px; }}
.muted {{ opacity:.68; }}
.empty {{ padding:40px; text-align:center; }}
@media (max-width:600px) {{ body{{padding:10px}} .stage{{min-height:300px}} }}
</style>
</head>
<body><main>
<h1>{safe_title}</h1>
<div class="controls">
 <label>Method<select id="method"></select></label>
 <label>N<select id="n"></select></label>
 <label>Seed<select id="seed"></select></label>
 <label>Speed<select id="speed"><option value="1200">0.5×</option><option value="600" selected>1×</option><option value="300">2×</option><option value="150">4×</option></select></label>
 <button id="play" type="button">Play</button>
 <button id="restart" type="button">Restart</button>
</div>
<div class="timeline"><strong id="round">Initial</strong><input id="slider" type="range" min="0" value="0"><span id="frame"></span></div>
<div class="status" id="status"></div>
<div class="stage" id="stage"><svg id="graph" viewBox="0 0 1000 650" role="img" aria-label="Topology evolution"></svg></div>
<div class="legend"><span><i class="swatch" style="border-color:#19a55a"></i>added this update</span><span><i class="swatch" style="border-color:#d64545"></i>removed this update</span><span class="muted">Layout is fixed for each run.</span></div>
</main>
<script>
const runs={payload};
const $=id=>document.getElementById(id);
const method=$('method'), nsel=$('n'), seed=$('seed'), slider=$('slider'), graph=$('graph');
let current=null, timer=null;
const uniq=a=>[...new Set(a)];
function options(el,values,keep){{el.innerHTML=''; values.forEach(v=>{{let o=document.createElement('option');o.value=v;o.textContent=v;el.appendChild(o)}});if(values.map(String).includes(String(keep)))el.value=keep;}}
function refreshMethods(){{options(method,uniq(runs.map(r=>r.m)).sort(),method.value);refreshN();}}
function refreshN(){{let rows=runs.filter(r=>r.m===method.value);options(nsel,uniq(rows.map(r=>r.n)).sort((a,b)=>a-b),nsel.value);refreshSeed();}}
function refreshSeed(){{let rows=runs.filter(r=>r.m===method.value&&String(r.n)===nsel.value);options(seed,uniq(rows.map(r=>r.s)).sort((a,b)=>a-b),seed.value);selectRun();}}
function edgeKey(e){{return e[0]<e[1]?e[0]+'-'+e[1]:e[1]+'-'+e[0];}}
function stateAt(index){{let edges=new Map(current.initial.map(e=>[edgeKey(e),e]));for(let i=0;i<index;i++){{for(const e of current.frames[i].d)edges.delete(edgeKey(e));for(const e of current.frames[i].a)edges.set(edgeKey(e),e);}}return edges;}}
function line(e,cls){{let a=point(e[0]),b=point(e[1]);return `<line class="${{cls}}" x1="${{a[0]}}" y1="${{a[1]}}" x2="${{b[0]}}" y2="${{b[1]}}"/>`;}}
function point(i){{let p=current.positions[i];return [45+(p[0]+1)*455,45+(p[1]+1)*280];}}
function draw(){{if(!current)return;let index=+slider.value, edges=stateAt(index), change=index?current.frames[index-1]:{{a:[],d:[]}};let added=new Set(change.a.map(edgeKey));let base=[...edges.values()].filter(e=>!added.has(edgeKey(e)));let markup=base.map(e=>line(e,'base')).join('')+change.a.map(e=>line(e,'added')).join('')+change.d.map(e=>line(e,'removed')).join('');let radius=current.n<=100?4.2:current.n<=200?3:2;markup+=current.positions.map((_,i)=>{{let p=point(i);return `<circle class="node" cx="${{p[0]}}" cy="${{p[1]}}" r="${{radius}}"><title>client ${{i}}</title></circle>`}}).join('');graph.innerHTML=markup;let label=index?`Round ${{change.r}}`:'Initial';$('round').textContent=label;$('frame').textContent=`frame ${{index}}/${{current.frames.length}}`;$('status').innerHTML=`<span><strong>${{current.m}}</strong></span><span>N=${{current.n}}</span><span>seed=${{current.s}}</span><span>edges=${{edges.size}}</span><span>+${{change.a.length}} / −${{change.d.length}}</span><span>Dmax=${{current.dmax}}</span><span>${{current.complete?'complete':'partial'}}</span>`;}}
function selectRun(){{stop();current=runs.find(r=>r.m===method.value&&String(r.n)===nsel.value&&String(r.s)===seed.value);if(!current){{graph.innerHTML='<text x="500" y="325" text-anchor="middle">No exact edge-delta data</text>';return;}}slider.max=current.frames.length;slider.value=0;draw();}}
function stop(){{if(timer)clearInterval(timer);timer=null;$('play').textContent='Play';}}
function play(){{if(timer){{stop();return;}}if(+slider.value>=+slider.max)slider.value=0;$('play').textContent='Pause';timer=setInterval(()=>{{if(+slider.value>=+slider.max){{stop();return;}}slider.value=+slider.value+1;draw();}},+$('speed').value);}}
method.onchange=refreshN;nsel.onchange=refreshSeed;seed.onchange=selectRun;slider.oninput=()=>{{stop();draw();}};$('play').onclick=play;$('restart').onclick=()=>{{stop();slider.value=0;draw();}};$('speed').onchange=()=>{{if(timer){{stop();play();}}}};
if(runs.length)refreshMethods();else $('stage').innerHTML='<div class="empty">No PAC topology-delta runs found.</div>';
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, required=True, help="Output directory to scan recursively")
    parser.add_argument("--output", type=Path, required=True, help="Destination HTML file")
    parser.add_argument("--methods", nargs="*", help="Optional method allow-list")
    parser.add_argument("--title", default="LFHE topology evolution")
    args = parser.parse_args()
    runs = collect_runs(args.outputs, set(args.methods) if args.methods else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(runs, args.title), encoding="utf-8")
    print(f"wrote {args.output} with {len(runs)} selectable runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
