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
    cmd=[sys.executable,args.main,"--method",r["method"],"--num-clients",r["num_clients"],
         "--seed",r["seed"],"--rounds",r["rounds"],"--protocol",r["protocol"],
         "--alpha",r["alpha"],"--dmax",r["dmax"],"--topology-interval",r["topology_interval"],
         "--eval-interval",r["eval_interval"],"--initial-graph",r["initial_graph"],
         "--participation-rate",r["participation_rate"],"--local-epochs",r["local_epochs"],
         "--batch-size",r["batch_size"],"--checkpoint-interval",r["checkpoint_interval"],
         "--eval-clients",r["eval_clients"],"--update-mode",r["update_mode"],
         "--data-regime",r["data_regime"],"--link-failure-rate",r["link_failure_rate"],
         "--stale-view-rounds",r["stale_view_rounds"],"--representation-mode",r["representation_mode"],
         "--lfhe-start-round",r["lfhe_start_round"],"--output-dir",str(out),"--data-root",args.data_root]
    if r["final_eval_all"].lower()=="true": cmd.append("--final-eval-all")
    else: cmd.append("--no-final-eval-all")
    if r["samples_per_client"]: cmd += ["--samples-per-client",r["samples_per_client"]]
    if r["min_samples_per_client"]: cmd += ["--min-samples-per-client",r["min_samples_per_client"]]
    if (out/"checkpoint.pt").exists(): cmd.append("--resume")
    print("[exec]"," ".join(cmd),flush=True)
    return subprocess.call(cmd)
if __name__=="__main__": raise SystemExit(main())
