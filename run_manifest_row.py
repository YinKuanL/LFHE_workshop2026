#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, subprocess, sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--index",type=int,required=True)
    ap.add_argument("--main",default="main.py"); ap.add_argument("--data-root",default="./data")
    args=ap.parse_args()
    with open(args.manifest,newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    if not 0<=args.index<len(rows): raise SystemExit(f"index {args.index} outside 0..{len(rows)-1}")
    r=rows[args.index]; out=Path(r["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    if (out/"SUCCESS").exists():
        print(f"[skip] {out}/SUCCESS exists",flush=True); return 0
    cmd=[sys.executable,args.main,"--output-dir",str(out),"--data-root",args.data_root]
    fields={"method":"--method","num_clients":"--num-clients","seed":"--seed","rounds":"--rounds",
      "protocol":"--protocol","alpha":"--alpha","dmax":"--dmax","degree_regime":"--degree-regime",
      "topology_interval":"--topology-interval","eval_interval":"--eval-interval","initial_graph":"--initial-graph",
      "participation_rate":"--participation-rate","local_epochs":"--local-epochs","batch_size":"--batch-size",
      "lr":"--lr","checkpoint_interval":"--checkpoint-interval","checkpoint_policy":"--checkpoint-policy",
      "eval_clients":"--eval-clients","update_mode":"--update-mode","data_regime":"--data-regime",
      "link_failure_rate":"--link-failure-rate","stale_view_rounds":"--stale-view-rounds",
      "representation_mode":"--representation-mode","lfhe_start_round":"--lfhe-start-round"}
    for key,flag in fields.items():
        if r.get(key,"")!="": cmd += [flag,r[key]]
    if r["final_eval_all"].lower()=="true": cmd.append("--final-eval-all")
    else: cmd.append("--no-final-eval-all")
    if r["samples_per_client"]: cmd += ["--samples-per-client",r["samples_per_client"]]
    if r["min_samples_per_client"]: cmd += ["--min-samples-per-client",r["min_samples_per_client"]]
    if r.get("checkpoint_policy","auto")!="disabled" and (out/"checkpoint.pt").exists(): cmd.append("--resume")
    print("[exec]"," ".join(cmd),flush=True)
    return subprocess.call(cmd)
if __name__=="__main__": raise SystemExit(main())
