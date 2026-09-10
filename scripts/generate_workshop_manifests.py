#!/usr/bin/env python3
"""Deterministically generate every workshop scalability manifest."""
from __future__ import annotations
import argparse,csv,io,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FIELDS=("suite","method","num_clients","seed","rounds","protocol","alpha","dmax","degree_regime",
 "topology_interval","eval_interval","initial_graph","participation_rate","local_epochs","batch_size","lr",
 "checkpoint_interval","checkpoint_policy","eval_clients","final_eval_all","update_mode","data_regime",
 "samples_per_client","min_samples_per_client","link_failure_rate","stale_view_rounds","representation_mode",
 "lfhe_start_round","output_dir")
SEEDS3=(42,43,44); SEEDS5=(42,43,44,45,46)
EXPECTED={"workshop_headline":120,"workshop_large_scale":12,"workshop_large_scale_feasibility":12,
 "workshop_degree_sweep":99,"workshop_fixed_per_client":27,"workshop_alpha_sweep":27,
 "workshop_concurrency":54,"workshop_participation":36,"workshop_link_failure":27,
 "workshop_combined_stress":9,"workshop_hard_topologies":24}

def row(suite,method,n,seed,root,**kw):
    value={"suite":suite,"method":method,"num_clients":n,"seed":seed,"rounds":300,"protocol":"scalable",
      "alpha":.3,"dmax":4,"degree_regime":"fixed4","topology_interval":5,"eval_interval":5,
      "initial_graph":"bounded_connected","participation_rate":1.0,"local_epochs":1,"batch_size":32,"lr":.05,
      "checkpoint_interval":10,"checkpoint_policy":"disabled" if method=="morph" and n>=500 else "auto",
      "eval_clients":50,"final_eval_all":"true","update_mode":"sequential","data_regime":"fixed_total",
      "samples_per_client":"","min_samples_per_client":"","link_failure_rate":0,"stale_view_rounds":0,
      "representation_mode":"flatten","lfhe_start_round":0}
    value.update(kw)
    tags=[method,f"n{n}",f"seed{seed}","initheadroomv2",f"d{value['dmax']}",str(value['degree_regime']),value['initial_graph'],
      value['update_mode'],f"stale{value['stale_view_rounds']}",f"p{value['participation_rate']}",
      f"link{value['link_failure_rate']}",value['data_regime'],f"a{value['alpha']}"]
    if value["samples_per_client"]!="": tags.append(f"samples{value['samples_per_client']}")
    value["output_dir"]=f"outputs/{root}/"+"_".join(map(str,tags))
    return value

def generate():
    suites={name:[] for name in EXPECTED}
    methods=("static_random","epidemic","dissdl","random_fof","morph","lfhe")
    for n in (50,100,200,500):
      for method in methods:
       for seed in SEEDS5: suites["workshop_headline"].append(row("workshop_headline",method,n,seed,"workshop_headline_d4_initheadroomv2"))
    for method in ("static_random","random_fof","lfhe","morph"):
      for seed in SEEDS3:
       suites["workshop_large_scale"].append(row("workshop_large_scale",method,1000,seed,"workshop_large_scale"))
       suites["workshop_large_scale_feasibility"].append(row("workshop_large_scale_feasibility",method,1000,seed,"workshop_large_scale_feasibility",rounds=5))
    for n in (100,200,500):
      regimes=[("fixed2",2),("fixed4",4),("fixed8",8)]
      logd=math.ceil(math.log2(n))
      if logd not in {x[1] for x in regimes}: regimes.append(("log2",logd))
      for degree_regime,dmax in regimes:
       for method in ("static_random","random_fof","lfhe"):
        for seed in SEEDS3: suites["workshop_degree_sweep"].append(row("workshop_degree_sweep",method,n,seed,"workshop_degree_sweep",dmax=dmax,degree_regime=degree_regime))
    for n in (100,200,500):
      for method in ("static_random","random_fof","lfhe"):
       for seed in SEEDS3: suites["workshop_fixed_per_client"].append(row("workshop_fixed_per_client",method,n,seed,"workshop_fixed_per_client",data_regime="fixed_samples_per_client",samples_per_client=100))
    for alpha in (.1,.3,1.0):
      for method in ("static_random","random_fof","lfhe"):
       for seed in SEEDS3: suites["workshop_alpha_sweep"].append(row("workshop_alpha_sweep",method,200,seed,"workshop_alpha_sweep",alpha=alpha))
    for n in (100,200,500):
      for mode in ("sequential","snapshot_concurrent"):
       for stale in (0,1,3):
        for seed in SEEDS3: suites["workshop_concurrency"].append(row("workshop_concurrency","lfhe",n,seed,"workshop_concurrency",update_mode=mode,stale_view_rounds=stale))
    for rate in (1.0,.9,.7,.5):
      for method in ("static_random","random_fof","lfhe"):
       for seed in SEEDS3: suites["workshop_participation"].append(row("workshop_participation",method,200,seed,"workshop_participation",participation_rate=rate))
    for rate in (0,.1,.3):
      for method in ("static_random","random_fof","lfhe"):
       for seed in SEEDS3: suites["workshop_link_failure"].append(row("workshop_link_failure",method,200,seed,"workshop_link_failure",link_failure_rate=rate))
    for method in ("static_random","random_fof","lfhe"):
      for seed in SEEDS3: suites["workshop_combined_stress"].append(row("workshop_combined_stress",method,200,seed,"workshop_combined_stress",participation_rate=.8,link_failure_rate=.1,stale_view_rounds=1))
    for graph in ("bounded_connected","ring","clustered","disconnected_clusters"):
      for method in ("random_fof","lfhe"):
       for seed in SEEDS3: suites["workshop_hard_topologies"].append(row("workshop_hard_topologies",method,200,seed,"workshop_hard_topologies",initial_graph=graph))
    return suites

def text(rows):
    stream=io.StringIO(newline=""); writer=csv.DictWriter(stream,fieldnames=FIELDS,lineterminator="\n"); writer.writeheader(); writer.writerows(rows); return stream.getvalue()

def validate(suites):
    seen=set()
    for name,rows in suites.items():
        if len(rows)!=EXPECTED[name]: raise ValueError(f"{name}: {len(rows)} != {EXPECTED[name]}")
        for value in rows:
            if value["method"] not in ("static_random","epidemic","dissdl","random_fof","morph","lfhe"): raise ValueError(value)
            if int(value["dmax"])<2 or int(value["num_clients"])<=int(value["dmax"]): raise ValueError(value)
            if value["output_dir"] in seen: raise ValueError(f"duplicate output: {value['output_dir']}")
            if "paper_exact" in value["output_dir"]: raise ValueError(value)
            seen.add(value["output_dir"])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--force",action="store_true"); ap.add_argument("--check",action="store_true"); args=ap.parse_args()
    suites=generate(); validate(suites)
    for name,rows in suites.items():
        path=ROOT/"manifests"/f"{name}.csv"; expected=text(rows)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8-sig")!=expected: raise SystemExit(f"out of date: {path}")
        elif path.exists() and path.read_text(encoding="utf-8-sig")!=expected and not args.force:
            raise SystemExit(f"refusing to overwrite modified manifest without --force: {path}")
        else: path.write_text(expected,encoding="utf-8",newline="")
        print(f"{name}: {len(rows)}")
if __name__=="__main__": main()
