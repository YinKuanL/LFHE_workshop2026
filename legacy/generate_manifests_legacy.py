#!/usr/bin/env python3
"""Generate approval-gated workshop manifests; this script never submits jobs."""
import csv, itertools
from pathlib import Path

DEST=Path(__file__).parent/"manifests"; DEST.mkdir(exist_ok=True)
FIELDS=["method","num_clients","seed","rounds","protocol","alpha","dmax","topology_interval","eval_interval","initial_graph","participation_rate","local_epochs","local_steps","batch_size","checkpoint_interval","eval_clients","final_eval_all","update_mode","data_regime","samples_per_client","min_samples_per_client","link_failure_rate","stale_view_rounds","output_dir"]
def write(name,rows):
    with (DEST/name).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
def row(group,m,n,s,rounds=300,d="4",part=1.,initial="bounded_connected",update="sequential",regime="fixed_total",samples="",minimum=None,link=0.,stale=0):
    if minimum is None: minimum=1 if group.startswith("stage01") else (10 if n<=200 else 5 if n<=500 else 3 if n<=1000 else 2)
    protocol="canonical" if group.startswith("stage01") else "scalable"
    if protocol=="canonical": initial="canonical_er"; minimum=1
    suffix=f"{m}_n{n}_seed{s}_d{d}_p{part:g}_{initial}_{update}_stale{stale}_link{link:g}_{regime}"
    return dict(method=m,num_clients=n,seed=s,rounds=rounds,protocol=protocol,alpha=.1,dmax=d,topology_interval=5,eval_interval=5,initial_graph=initial,participation_rate=part,local_epochs=1,local_steps="",batch_size=32,checkpoint_interval=10 if n<1000 else 25,eval_clients=min(50,n),final_eval_all="true",update_mode=update,data_regime=regime,samples_per_client=samples,min_samples_per_client=minimum,link_failure_rate=link,stale_view_rounds=stale,output_dir=f"outputs/{group}/{suffix}")

# Gate 1: no downstream stage is approved until these match historical ranges.
write("stage01_canonical_alignment.csv",[row("stage01_canonical",m,30,s) for m,s in itertools.product(("ring","static_random","epidemic","dissdl","lfhe"),range(42,47))])

# Gate 2/3: mandatory scale pilots. Five-round summaries are reviewed before 20-round runs.
pilot=lambda group,rounds,ns:[row(group,m,n,42,rounds=rounds,part=p) for n,m,p in itertools.product(ns,("static_random","lfhe"),(1.,.1))]
write("stage02_feasibility_5round.csv",pilot("stage02_feas5",5,(100,500,1000)))
write("stage03_feasibility_20round.csv",pilot("stage03_feas20",20,(100,500,1000)))
write("optional_feasibility_1500_2000_5round.csv",pilot("optional_feas5",5,(1500,2000)))
write("optional_feasibility_1500_2000_20round.csv",pilot("optional_feas20",20,(1500,2000)))

# Primary fixed-degree headline, first three seeds; extra seeds require review.
primary=lambda group,seeds:[row(group,m,n,s) for n,m,s in itertools.product((100,200,500,1000),("static_random","random_fof","lfhe"),seeds)]
write("stage04_primary_fixed_d4_seeds42_44.csv",primary("stage04_primary",(42,43,44)))
write("optional_headline_seeds45_46.csv",primary("optional_headline",(45,46)))

# Degree scaling is separate from the D_max=4 headline.
write("stage05_degree_sweep_500_1000.csv",[row("stage05_degree",m,n,s,d=d) for n,m,d,s in itertools.product((500,1000),("static_random","random_fof","lfhe"),("2","4","8","log2"),(42,43,44))])
write("optional_degree_sweep_n2000.csv",[row("optional_degree2000",m,2000,s,d=d) for m,d,s in itertools.product(("static_random","random_fof","lfhe"),("2","4","8","log2"),(42,43,44))])

# Workshop-specific concurrency: hard clustered topology plus stale representation views.
concurrency=[]
for n,initial,s in itertools.product((100,500),("bounded_connected","clustered"),(42,43,44)):
    concurrency.append(row("stage06_concurrency","lfhe",n,s,initial=initial,update="sequential"))
    concurrency.append(row("stage06_concurrency","lfhe",n,s,initial=initial,update="snapshot_concurrent"))
    concurrency.append(row("stage06_concurrency","lfhe",n,s,initial=initial,update="snapshot_concurrent",stale=5))
write("stage06_concurrent_and_stale_views.csv",concurrency)

# Explicit FoF disconnected-component limitation demonstration.
write("stage07_fof_disconnected_limit.csv",[row("stage07_disconnected",m,100,s,initial="disconnected_clusters") for m,s in itertools.product(("random_fof","lfhe"),(42,43,44))])

# Link-failure stress; rate=0 controls are reused from the primary stage.
write("stage08_link_failure_stress.csv",[row("stage08_link_failure",m,500,s,link=rate) for m,rate,s in itertools.product(("static_random","random_fof","lfhe"),(.1,.3),(42,43,44))])

# Second data regime: constant 25 samples/client, so changing N does not reduce local sample count.
write("stage09_fixed_samples_per_client.csv",[row("stage09_fixed_per_client",m,n,s,regime="fixed_per_client",samples=25,minimum=25) for n,m,s in itertools.product((100,500,1000),("static_random","random_fof","lfhe"),(42,43,44))])
