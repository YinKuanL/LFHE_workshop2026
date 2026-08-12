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
.stage {{ border:1px solid color-mix(in srgb, CanvasText 22%, transparent); min-height:520px; background:color-mix(in srgb, Canvas 96%, CanvasText 4%); }}
svg {{ width:100%; height:auto; display:block; }}
.base {{ stroke:color-mix(in srgb, CanvasText 38%, transparent); stroke-width:1; }}
.added {{ stroke:#19a55a; stroke-width:3; }}
.removed {{ stroke:#d64545; stroke-width:3; stroke-dasharray:6 4; }}
.edge {{ vector-effect:non-scaling-stroke; }}
.node {{ fill:Canvas; stroke:CanvasText; stroke-width:1.25; vector-effect:non-scaling-stroke; }}
.node.changed {{ fill:#f2b84b; }}
.node.focus {{ fill:#4b9cf2; stroke-width:2.5; }}
.node-label {{ fill:CanvasText; font-size:5.5px; paint-order:stroke; stroke:Canvas; stroke-width:2px; stroke-linejoin:round; }}
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
 <label>View<select id="view"><option value="neighborhood">Client neighborhood</option><option value="overview">Full overview</option></select></label>
 <label>Display<select id="display"><option value="topology">Topology only</option><option value="changes">Highlight changes</option></select></label>
 <label>Client<select id="client"></select></label>
 <label>Hops<select id="hops"><option value="1">1 hop</option><option value="2" selected>2 hops</option><option value="3">3 hops</option></select></label>
 <label>Speed<select id="speed"><option value="1200">0.5×</option><option value="600" selected>1×</option><option value="300">2×</option><option value="150">4×</option></select></label>
 <button id="play" type="button">Play</button>
 <button id="restart" type="button">Restart</button>
</div>
<div class="timeline"><strong id="round">Initial</strong><input id="slider" type="range" min="0" value="0"><span id="frame"></span></div>
<div class="status" id="status"></div>
<div class="stage" id="stage"><svg id="graph" viewBox="0 0 1000 650" role="img" aria-label="Topology evolution"></svg></div>
<div class="legend"><span id="added-legend"><i class="swatch" style="border-color:#19a55a"></i>added this update</span><span id="removed-legend"><i class="swatch" style="border-color:#d64545"></i>removed this update</span><span>blue node = selected client</span><span class="muted">Use Full overview for the global shape.</span></div>
</main>
<script>
const runs={payload};
const $=id=>document.getElementById(id);
const method=$('method'), nsel=$('n'), seed=$('seed'), view=$('view'), display=$('display'), client=$('client'), hops=$('hops'), slider=$('slider'), graph=$('graph');
let current=null, timer=null;
const uniq=a=>[...new Set(a)];
function options(el,values,keep){{el.innerHTML=''; values.forEach(v=>{{let o=document.createElement('option');o.value=v;o.textContent=v;el.appendChild(o)}});if(values.map(String).includes(String(keep)))el.value=keep;}}
function refreshMethods(){{options(method,uniq(runs.map(r=>r.m)).sort(),method.value);refreshN();}}
function refreshN(){{let rows=runs.filter(r=>r.m===method.value);options(nsel,uniq(rows.map(r=>r.n)).sort((a,b)=>a-b),nsel.value);refreshSeed();}}
function refreshSeed(){{let rows=runs.filter(r=>r.m===method.value&&String(r.n)===nsel.value);options(seed,uniq(rows.map(r=>r.s)).sort((a,b)=>a-b),seed.value);selectRun();}}
function edgeKey(e){{return e[0]<e[1]?e[0]+'-'+e[1]:e[1]+'-'+e[0];}}
function stateAt(index){{let edges=new Map(current.initial.map(e=>[edgeKey(e),e]));for(let i=0;i<index;i++){{for(const e of current.frames[i].d)edges.delete(edgeKey(e));for(const e of current.frames[i].a)edges.set(edgeKey(e),e);}}return edges;}}
function line(e,cls){{let a=point(e[0]),b=point(e[1]);return `<line class="edge ${{cls}}" x1="${{a[0]}}" y1="${{a[1]}}" x2="${{b[0]}}" y2="${{b[1]}}"/>`;}}
function point(i){{let p=current.positions[i];return [45+(p[0]+1)*455,45+(p[1]+1)*280];}}
function visibleNodes(edges,change){{if(view.value==='overview')return new Set(current.positions.map((_,i)=>i));let focus=+client.value,adj=Array.from({{length:current.n}},()=>[]);for(const e of edges.values()){{adj[e[0]].push(e[1]);adj[e[1]].push(e[0]);}}let visible=new Set([focus]),frontier=[focus];for(let step=0;step<+hops.value;step++){{let next=[];for(const node of frontier)for(const peer of adj[node])if(!visible.has(peer)){{visible.add(peer);next.push(peer);}}frontier=next;}}if(display.value==='changes')for(const e of [...change.a,...change.d])if(visible.has(e[0])||visible.has(e[1])){{visible.add(e[0]);visible.add(e[1]);}}return visible;}}
function fittedViewBox(visible){{if(view.value==='overview')return '0 0 1000 650';let points=[...visible].map(point),xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),w=Math.max(220,xmax-xmin+90),h=Math.max(145,ymax-ymin+90),ratio=1000/650;if(w/h>ratio)h=w/ratio;else w=h*ratio;let cx=(xmin+xmax)/2,cy=(ymin+ymax)/2;return `${{cx-w/2}} ${{cy-h/2}} ${{w}} ${{h}}`;}}
function draw(){{if(!current)return;let index=+slider.value,edges=stateAt(index),change=index?current.frames[index-1]:{{a:[],d:[]}},visible=visibleNodes(edges,change),showChanges=display.value==='changes',added=new Set(showChanges?change.a.map(edgeKey):[]),changed=new Set(showChanges?[...change.a,...change.d].flat():[]),base=[...edges.values()].filter(e=>!added.has(edgeKey(e))&&visible.has(e[0])&&visible.has(e[1])),adds=showChanges?change.a.filter(e=>visible.has(e[0])&&visible.has(e[1])):[],removes=showChanges?change.d.filter(e=>visible.has(e[0])&&visible.has(e[1])):[];graph.setAttribute('viewBox',fittedViewBox(visible));let markup=base.map(e=>line(e,'base')).join('')+removes.map(e=>line(e,'removed')).join('')+adds.map(e=>line(e,'added')).join('');let focus=+client.value,showAllLabels=view.value==='neighborhood'&&visible.size<=35;for(const i of visible){{let p=point(i),isFocus=view.value==='neighborhood'&&i===focus,classes=`node${{changed.has(i)?' changed':''}}${{isFocus?' focus':''}}`,radius=isFocus?4.4:view.value==='overview'?(current.n<=100?4:current.n<=200?3:2):2.8;markup+=`<circle class="${{classes}}" cx="${{p[0]}}" cy="${{p[1]}}" r="${{radius}}"><title>client ${{i}}</title></circle>`;if(showAllLabels||isFocus||changed.has(i))markup+=`<text class="node-label" x="${{p[0]+4.5}}" y="${{p[1]-4.5}}">${{i}}</text>`;}}graph.innerHTML=markup;$('added-legend').hidden=!showChanges;$('removed-legend').hidden=!showChanges;let label=index?`Round ${{change.r}}`:'Initial';$('round').textContent=label;$('frame').textContent=`frame ${{index}}/${{current.frames.length}}`;$('status').innerHTML=`<span><strong>${{current.m}}</strong></span><span>N=${{current.n}}</span><span>seed=${{current.s}}</span><span>edges=${{edges.size}}</span><span>visible=${{visible.size}}</span>${{showChanges?`<span>+${{change.a.length}} / −${{change.d.length}}</span>`:''}}<span>Dmax=${{current.dmax}}</span><span>${{current.complete?'complete':'partial'}}</span>`;}}
function selectRun(){{stop();current=runs.find(r=>r.m===method.value&&String(r.n)===nsel.value&&String(r.s)===seed.value);if(!current){{graph.innerHTML='<text x="500" y="325" text-anchor="middle">No exact edge-delta data</text>';return;}}let preferred=current.frames[0]?.a[0]?.[0]??current.frames[0]?.d[0]?.[0]??0;options(client,current.positions.map((_,i)=>i),preferred);view.value=current.n>100?'neighborhood':'overview';client.disabled=view.value==='overview';hops.disabled=view.value==='overview';slider.max=current.frames.length;slider.value=0;draw();}}
function stop(){{if(timer)clearInterval(timer);timer=null;$('play').textContent='Play';}}
function play(){{if(timer){{stop();return;}}if(+slider.value>=+slider.max)slider.value=0;$('play').textContent='Pause';timer=setInterval(()=>{{if(+slider.value>=+slider.max){{stop();return;}}slider.value=+slider.value+1;draw();}},+$('speed').value);}}
method.onchange=refreshN;nsel.onchange=refreshSeed;seed.onchange=selectRun;view.onchange=()=>{{client.disabled=view.value==='overview';hops.disabled=view.value==='overview';draw();}};display.onchange=draw;client.onchange=draw;hops.onchange=draw;slider.oninput=()=>{{stop();draw();}};$('play').onclick=play;$('restart').onclick=()=>{{stop();slider.value=0;draw();}};$('speed').onchange=()=>{{if(timer){{stop();play();}}}};
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
