#!/usr/bin/env python3
"""Regenerate the checked-in experiment manifests deterministically."""
import csv, itertools, math
from pathlib import Path

ROOT=Path(__file__).parent; DEST=ROOT/"manifests"; DEST.mkdir(exist_ok=True)
FIELDS=["method","num_clients","seed","rounds","protocol","alpha","dmax","topology_interval","eval_interval","initial_graph","participation_rate","local_epochs","local_steps","batch_size","checkpoint_interval","eval_clients","final_eval_all","update_mode","output_dir"]
def write(name, rows):
    with (DEST/name).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
def row(group,m,n,s,d="4",participation=1.0,protocol="scalable",initial="bounded_connected",checkpoint=10):
    return dict(method=m,num_clients=n,seed=s,rounds=300,protocol=protocol,alpha=.1,dmax=d,topology_interval=5,eval_interval=5,initial_graph=initial,participation_rate=participation,local_epochs=1,local_steps="",batch_size=32,checkpoint_interval=checkpoint,eval_clients=min(50,n),final_eval_all="true",update_mode="sequential",output_dir=f"outputs/{group}/{m}_n{n}_seed{s}_dmax{d}")
write("canonical_alignment.csv",[row("canonical",m,30,s,protocol="canonical",initial="canonical_er") for m,s in itertools.product(("ring","static_random","epidemic","dissdl","lfhe"),range(42,47))])
write("fixed_degree_scaling_pilot.csv",[row("scaling_pilot",m,n,s) for m,n,s in itertools.product(("ring","static_random","epidemic","dissdl","random_fof","lfhe"),(25,50,100,200,500),(42,43,44))])
write("n1000_feasibility.csv",[row("n1000",m,1000,42,participation=.1,checkpoint=25) for m in ("static_random","random_fof","lfhe")])
write("degree_sweep.csv",[row("degree_sweep",m,n,s,d) for m,n,s,d in itertools.product(("static_random","random_fof","lfhe"),(100,200,500),(42,43,44),("2","4","8","log2"))])
