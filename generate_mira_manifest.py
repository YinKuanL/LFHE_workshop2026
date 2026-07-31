#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, math
from pathlib import Path

FIELDS = [
    "stage","method","num_clients","seed","rounds","protocol","alpha","dmax",
    "topology_interval","eval_interval","initial_graph","participation_rate",
    "local_epochs","batch_size","checkpoint_interval","eval_clients","final_eval_all",
    "update_mode","data_regime","samples_per_client","min_samples_per_client",
    "link_failure_rate","stale_view_rounds","representation_mode","lfhe_start_round","output_dir"
]

def row(stage, method, n, seed, *, rounds=300, protocol="scalable", alpha=.1, dmax="4",
        ti=5, ei=5, graph="bounded_connected", rate=1.0, epochs=1, batch=32,
        checkpoint=10, eval_clients=50, final_all=True, update="sequential",
        regime="fixed_total", samples="", minimum="", link=0.0, stale=0,
        representation="flatten", start=0):
    suffix=(f"{method}_n{n}_seed{seed}_d{dmax}_p{rate:g}_{graph}_{representation}"
            f"_start{start}_link{link:g}_stale{stale}_{regime}")
    return dict(stage=stage,method=method,num_clients=n,seed=seed,rounds=rounds,
        protocol=protocol,alpha=alpha,dmax=dmax,topology_interval=ti,eval_interval=ei,
        initial_graph=graph,participation_rate=rate,local_epochs=epochs,batch_size=batch,
        checkpoint_interval=checkpoint,eval_clients=eval_clients,final_eval_all=str(final_all).lower(),
        update_mode=update,data_regime=regime,samples_per_client=samples,
        min_samples_per_client=minimum,link_failure_rate=link,stale_view_rounds=stale,
        representation_mode=representation,lfhe_start_round=start,
        output_dir=f"outputs/{stage}/{suffix}")

def build(level: str):
    rows=[]; seeds=(42,43,44); canonical_seeds=(42,43,44,45,46)
    # Main-paper reproduction: exact old setting, plus Morph.
    for n in (10,50,100,500):
        d=max(2,int(2*math.log(n)))
        for method in ("ring","static_random","epidemic","dissdl","morph","lfhe"):
            for seed in seeds:
                rows.append(row("paper_alignment",method,n,seed,alpha=.3,dmax=str(d),
                    graph="canonical_er",eval_clients=n,representation="class_mean",start=0))
    # Corrected canonical alignment used by the workshop implementation.
    for method in ("ring","static_random","epidemic","dissdl","random_fof","morph","lfhe"):
        for seed in canonical_seeds:
            rows.append(row("canonical",method,30,seed,protocol="canonical",alpha=.1,dmax="4",
                graph="canonical_er",eval_clients=30,representation="flatten",start=21))
    # Five-round feasibility gate up to 2000 clients.
    for n in (100,500,1000,1500,2000):
        for method in ("static_random","morph","lfhe"):
            rows.append(row("feasibility",method,n,42,rounds=5,checkpoint=5))
    # Primary fixed-degree scaling.
    for n in (100,200,500,1000,1500,2000):
        for method in ("static_random","random_fof","morph","lfhe"):
            for seed in seeds:
                rows.append(row("primary_d4",method,n,seed))
    if level == "core":
        return rows
    # Degree sweep.
    for n in (100,500,1000,2000):
        for d in ("2","4","8","log2"):
            for method in ("static_random","random_fof","morph","lfhe"):
                for seed in seeds:
                    rows.append(row("degree_sweep",method,n,seed,dmax=d))
    # Partial participation deployment.
    for n in (1000,1500,2000):
        for method in ("static_random","morph","lfhe"):
            for seed in seeds:
                rows.append(row("partial_p01",method,n,seed,rate=.1))
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--level",choices=("core","all"),default="all")
    ap.add_argument("--output",default="manifests/mira_all.csv"); args=ap.parse_args()
    rows=build(args.level); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} experiments to {out}")
if __name__=="__main__": main()
