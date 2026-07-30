#!/usr/bin/env python3
"""Canonical-compatible and scalable LFHE CIFAR-10 experiment runner."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random, signal, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from epidemic import build_epidemic_graph
from dissdl import DissDLState
from lfhe import lfhe_update

EXIT_REQUEUE = 99
STOP_SIGNAL = None
METHODS = ("ring", "static_random", "epidemic", "dissdl", "random_fof", "lfhe", "fedavg")

def _stop(signum, _frame):
    global STOP_SIGNAL
    STOP_SIGNAL = signum
signal.signal(signal.SIGTERM, _stop)
if hasattr(signal, "SIGUSR1"): signal.signal(signal.SIGUSR1, _stop)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(True),
            nn.Conv2d(32,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(True),
            nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(True),nn.MaxPool2d(2))
        self.avgpool = nn.AdaptiveAvgPool2d((4,4))
        self.classifier = nn.Sequential(nn.Flatten(),nn.Linear(1024,256),nn.ReLU(True),nn.Dropout(.3),nn.Linear(256,10))
    def forward(self,x): return self.classifier(self.avgpool(self.features(x)))
    def get_representation(self): return self.classifier[-1].weight.data.flatten()

@dataclass
class Config:
    method:str; num_clients:int; seed:int; rounds:int; protocol:str; alpha:float; dmax_spec:str; dmax:int
    topology_interval:int; eval_interval:int; initial_graph:str; participation_rate:float
    local_epochs:int|None; local_steps:int|None; batch_size:int; lr:float; output_dir:str
    checkpoint_interval:int; checkpoint_path:str; resume:bool; force:bool; eval_clients:int
    final_eval_all:bool; data_root:str; update_mode:str; min_samples_per_client:int
    target_accuracy:float; graph_metric_interval:int; dissdl_max_n:int; num_workers:int
    w1:float=1.; w2:float=1.; w3:float=.1; epsilon:float=.05
    dataset:str="cifar10"; data_regime:str="fixed_total"; optimizer:str="SGD"

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def clone_state(state): return {k:v.detach().cpu().clone() for k,v in state.items()}
def initial_states(n, seed):
    torch.manual_seed(seed); base=clone_state(CNN().state_dict())
    return [{k:v.clone() for k,v in base.items()} for _ in range(n)]

def load_cifar(root):
    from torchvision import datasets, transforms
    train_tf=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),transforms.Normalize([.4914,.4822,.4465],[.2023,.1994,.2010])])
    test_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.4914,.4822,.4465],[.2023,.1994,.2010])])
    return (datasets.CIFAR10(root,train=True,download=True,transform=train_tf),
            datasets.CIFAR10(root,train=False,download=True,transform=test_tf))

def dirichlet_split(labels, n, alpha, minimum, seed, preserve_original=False, max_attempts=1000):
    """Original classwise split, deterministically resampled until the minimum holds."""
    if minimum*n > len(labels): raise ValueError("min-samples-per-client exceeds fixed dataset size")
    for attempt in range(max_attempts):
        rng=np.random.RandomState(seed+attempt); result=[[] for _ in range(n)]
        for c in range(int(labels.max())+1):
            idx=np.where(labels==c)[0]; rng.shuffle(idx)
            cuts=(np.cumsum(rng.dirichlet(np.repeat(alpha,n)))*len(idx)).astype(int)[:-1]
            for i,part in enumerate(np.split(idx,cuts)): result[i].extend(part.tolist())
        if min(map(len,result)) >= minimum: return result
        if preserve_original and minimum == 0: return result
    # Stable repair is the scalable fallback when rejection sampling is improbable.
    while min(map(len,result)) < minimum:
        receiver=min(range(n),key=lambda i:(len(result[i]),i)); donor=max(range(n),key=lambda i:(len(result[i]),-i))
        result[receiver].append(result[donor].pop())
    return result

def partition_stats(labels,splits):
    ent=[]
    for split in splits:
        if not split: ent.append(0.); continue
        counts=np.bincount(labels[np.asarray(split)],minlength=int(labels.max())+1); p=counts[counts>0]/counts.sum()
        ent.append(float(-(p*np.log(p)).sum()))
    sizes=list(map(len,splits))
    return {"min_samples":min(sizes),"max_samples":max(sizes),"mean_samples":float(np.mean(sizes)),
            "empty_clients":sum(x==0 for x in sizes),"mean_label_entropy":float(np.mean(ent)),"client_label_entropy":ent}

def canonical_er(n,seed):
    p=4/(n-1); rng=np.random.RandomState(seed)
    while True:
        g=nx.erdos_renyi_graph(n,p,seed=int(rng.randint(0,1_000_000)))
        if nx.is_connected(g): return g

def bounded_connected(n,dmax,seed):
    if n<3 or dmax<2: raise ValueError("bounded_connected requires N>=3 and D_max>=2")
    rng=random.Random(seed); order=list(range(n)); rng.shuffle(order); g=nx.Graph(); g.add_nodes_from(range(n))
    for i,u in enumerate(order): g.add_edge(u,order[(i+1)%n])
    pairs=[(u,v) for u in range(n) for v in range(u+1,n)]; rng.shuffle(pairs)
    for u,v in pairs:
        if g.degree(u)<dmax and g.degree(v)<dmax and not g.has_edge(u,v): g.add_edge(u,v)
    assert nx.is_connected(g) and max(dict(g.degree()).values())<=dmax
    return g

def initial_graph(cfg):
    if cfg.method=="ring": return nx.cycle_graph(cfg.num_clients)
    return canonical_er(cfg.num_clients,cfg.seed) if cfg.initial_graph=="canonical_er" else bounded_connected(cfg.num_clients,cfg.dmax,cfg.seed)

def graph_stats(g, expensive=True):
    und=g.to_undirected(); deg=[d for _,d in und.degree()]; comps=nx.number_connected_components(und)
    out={"edges":g.number_of_edges(),"mean_degree":float(np.mean(deg)),"min_degree":min(deg),"max_degree":max(deg),
         "connected_components":comps,"clustering_coefficient":float(nx.average_clustering(und))}
    if comps==1:
        if len(g)<=300 and expensive: out["diameter"]=nx.diameter(und); out["effective_path_length"]=nx.average_shortest_path_length(und)
        else:
            nodes=sorted(und)[:min(32,len(und))]; paths=[]
            for u in nodes: paths.extend(nx.single_source_shortest_path_length(und,u).values())
            out["sampled_path_length"]=float(np.mean(paths))
        if expensive:
            try:
                import scipy.sparse.linalg as sla
                lap=nx.normalized_laplacian_matrix(und).astype(float); vals=sla.eigsh(lap,k=2,which="SM",return_eigenvectors=False)
                out["normalized_spectral_gap"]=float(sorted(vals)[1])
            except Exception as exc: out["spectral_metric_error"]=type(exc).__name__
    return out

def weighted_average(items):
    ref=items[-1][1]; out={}
    for key,t in ref.items():
        if t.is_floating_point():
            acc=torch.zeros_like(t); [acc.add_(s[key],alpha=float(w)) for w,s in items]; out[key]=acc
        else: out[key]=t.clone()
    return out

def aggregate(states,g,active):
    active=set(active); updates={}; transmissions=0
    for i in active:
        neighbors=[j for j in g.neighbors(i) if j in active]; values=[]; total=0.
        for j in neighbors:
            w=1/(1+max(g.degree(i),g.degree(j))); total+=w; values.append((w,states[j])); transmissions+=1
        if total>1+1e-7: raise RuntimeError("negative self-weight")
        values.append((1-total,states[i])); updates[i]=weighted_average(values)
    for i,s in updates.items(): states[i]=s
    return transmissions

def fedavg(states,active):
    avg=weighted_average([(1/len(active),states[i]) for i in active])
    for i in active: states[i]=clone_state(avg)
    return 2*len(active)

def epidemic_aggregate(states,g,active):
    active=set(active); updates={}; tx=0
    for i in active:
        incoming=[j for j in g.predecessors(i) if j in active]; tx+=len(incoming)
        updates[i]=weighted_average([(1/(len(incoming)+1),states[j]) for j in incoming]+[(1/(len(incoming)+1),states[i])])
    for i,s in updates.items(): states[i]=s
    return tx

class _RepresentationModel:
    def __init__(self,state): self.weight=state["classifier.4.weight"]
    def get_representation(self): return self.weight.flatten()
class Adapter:
    """Lightweight canonical LFHE interface; does not allocate another CNN."""
    def __init__(self,state): self.model=_RepresentationModel(state)

def fof_update(g, active, dmax, random_policy=False):
    g=g.copy(); trace=[]
    for i in active:
        event={"candidate_checks":0,"fitness_evaluations":0,"action":"no_candidate"}; ni=list(g.neighbors(i))
        if ni:
            j=random.choice(ni); candidates=[k for k in g.neighbors(j) if k!=i and not g.has_edge(i,k)]
            if candidates:
                k=random.choice(candidates); event["candidate_checks"]=1; event["action"]="rejected_proposal"
                if g.degree(i)<dmax and g.degree(k)<dmax:
                    # Random baseline ranks/accepts independently of model fitness.
                    if random.random()<.5: g.add_edge(i,k); event["action"]="accepted_addition"
                elif g.degree(k)<dmax:
                    feasible=[]
                    for x in ni:
                        event["candidate_checks"]+=1
                        if g.degree(x)>1 and set(g.neighbors(i))&set(g.neighbors(x)): feasible.append(x)
                    if feasible and random.random()<.5:
                        g.remove_edge(i,random.choice(feasible)); g.add_edge(i,k); event["action"]="accepted_swap"
        trace.append(event)
    return g,trace

def dissdl_aggregate(states, nodes, active):
    active=set(active); updates={}; transmissions=0
    for i in active:
        incoming=[j for j in nodes[i].wanted_senders if j in active]; transmissions+=len(incoming)
        updates[i]=weighted_average([(1/(len(incoming)+1),states[j]) for j in incoming]+[(1/(len(incoming)+1),states[i])])
        rep_i=states[i]["classifier.4.weight"].flatten().float()
        for j in incoming:
            rep_j=states[j]["classifier.4.weight"].flatten().float(); sim=float(torch.nn.functional.cosine_similarity(rep_i,rep_j,dim=0))
            nodes[i].similarity_history.setdefault(j,[]).append(sim); nodes[i].similarity_history[j]=nodes[i].similarity_history[j][-5:]
    for i,s in updates.items(): states[i]=s
    return transmissions

def dissdl_update(nodes, states, active):
    for i in active:
        current=set(nodes[i].wanted_senders); available=list(nodes[i].known_peers-current)
        if not available or len(current)<=1: continue
        known=[x for values in nodes[i].similarity_history.values() for x in values]; default=float(np.mean(known)) if known else 0.
        def sim(j):
            if j in nodes[i].similarity_history: return nodes[i].similarity_history[j][-1]
            return default
        add_scores=torch.tensor([-sim(j) for j in available]); remove=list(current); remove_scores=torch.tensor([sim(j) for j in remove])
        plus=available[int(torch.multinomial(torch.softmax(add_scores,0),1))]; minus=remove[int(torch.multinomial(torch.softmax(remove_scores,0),1))]
        nodes[i].wanted_senders.remove(minus); nodes[i].wanted_senders.add(plus)
    graph=nx.DiGraph(); graph.add_nodes_from(range(len(nodes)))
    for i,node in enumerate(nodes): graph.add_edges_from((j,i) for j in node.wanted_senders)
    return graph

def train_client(model,state,dataset,indices,cfg,device,loader_seed):
    model.load_state_dict(state); model.to(device); model.train(); opt=torch.optim.SGD(model.parameters(),lr=cfg.lr); loss=nn.CrossEntropyLoss()
    # Canonical mode deliberately uses the global torch RNG like the historical
    # persistent DataLoaders. Scalable mode isolates sampler order per client/round.
    gen=None if cfg.protocol=="canonical" else torch.Generator().manual_seed(loader_seed)
    loader=DataLoader(Subset(dataset,indices),batch_size=cfg.batch_size,shuffle=True,generator=gen,num_workers=0)
    steps=0
    for _ in range(cfg.local_epochs or 1):
        for x,y in loader:
            opt.zero_grad(set_to_none=True); z=loss(model(x.to(device)),y.to(device)); z.backward(); opt.step(); steps+=1
            if cfg.local_steps is not None and steps>=cfg.local_steps: break
        if cfg.local_steps is not None and steps>=cfg.local_steps: break
    result=clone_state(model.state_dict()); model.cpu(); return result

def evaluate(model,states,ids,loader,device):
    criterion=nn.CrossEntropyLoss(reduction="sum"); accs=[]; losses=[]
    for i in ids:
        model.load_state_dict(states[i]); model.to(device); model.eval(); correct=total=0; total_loss=0.
        with torch.inference_mode():
            for x,y in loader:
                y=y.to(device); out=model(x.to(device)); total_loss+=float(criterion(out,y)); correct+=int((out.argmax(1)==y).sum()); total+=len(y)
        accs.append(correct/total); losses.append(total_loss/total); model.cpu()
    return accs,losses

def model_metrics(states):
    reps=torch.stack([s["classifier.4.weight"].flatten().float() for s in states]); mean=reps.mean(0)
    return {"consensus_error":float(((reps-mean).norm(dim=1)**2).mean()),"representation_variance":float(reps.var(dim=0,unbiased=False).mean())}

def rng_state():
    value={"python":random.getstate(),"numpy":np.random.get_state(),"torch_cpu":torch.get_rng_state()}
    if torch.cuda.is_available(): value["torch_cuda"]=torch.cuda.get_rng_state_all()
    return value
def restore_rng(value):
    random.setstate(value["python"]); np.random.set_state(value["numpy"]); torch.set_rng_state(value["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in value: torch.cuda.set_rng_state_all(value["torch_cuda"])

def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(value,indent=2),encoding="utf-8"); os.replace(tmp,path)
def atomic_checkpoint(path,value):
    tmp=path.with_name("checkpoint.tmp"); torch.save(value,tmp); os.replace(tmp,path)
def append_jsonl(path,value):
    with path.open("a",encoding="utf-8") as f: f.write(json.dumps(value)+"\n"); f.flush(); os.fsync(f.fileno())
def write_edges(path,g): nx.write_edgelist(g,path,data=False)
def config_hash(cfg):
    d=asdict(cfg); [d.pop(k,None) for k in ("resume","force","checkpoint_path")]
    return hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()[:16]

def summarize(cfg, records, started, pstats, initial, final, deployment):
    evals=[r for r in records if "mean_accuracy" in r]; xs=[r["round"] for r in evals]; ys=[r["mean_accuracy"] for r in evals]
    auc=float(np.trapz(ys,xs)/(xs[-1]-xs[0])) if len(xs)>1 else (ys[0] if ys else None)
    reached=next((r for r in evals if r["mean_accuracy"]>=cfg.target_accuracy),None)
    accepted=[r["round"] for r in records if r.get("accepted_additions",0)+r.get("accepted_swaps",0)>0]
    return {"experiment_id":config_hash(cfg),"status":"complete","protocol":cfg.protocol,"deployment_protocol":deployment,
      "final_accuracy":ys[-1] if ys else None,"normalized_auc":auc,"rounds_to_target":reached["round"] if reached else None,
      "bytes_to_target":reached["cumulative_bytes"] if reached else None,"wall_clock_time_to_target":reached["elapsed_seconds"] if reached else None,
      "partition":pstats,"initial_graph":initial,"final_graph":final,"wall_clock_seconds":time.time()-started,
      "evaluations":len(evals),"rounds_completed":cfg.rounds,"last_accepted_rewire_round":accepted[-1] if accepted else None,
      "topology_stabilization_round":(accepted[-1]+1) if accepted else 0,
      "scientific_notes":["canonical LFHE fitness/annealing/representation" if cfg.method=="lfhe" else "matched baseline budget"]}

def run(cfg):
    set_seed(cfg.seed); out=Path(cfg.output_dir); success=out/"SUCCESS"
    if success.exists() and not cfg.force: print(f"[skip] {success} exists"); return 0
    if cfg.force and out.exists():
        for name in ("config.json","checkpoint.pt","checkpoint.tmp","metrics.jsonl","summary.json","graph_initial.edgelist","graph_final.edgelist","SUCCESS"):
            path=out/name
            if path.exists(): path.unlink()
    if out.exists() and any(out.iterdir()) and not cfg.resume and not cfg.force: raise ValueError("output-dir is non-empty; use --resume, --force, or a unique path")
    out.mkdir(parents=True,exist_ok=True); cp=Path(cfg.checkpoint_path) if cfg.checkpoint_path else out/"checkpoint.pt"
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); started=time.time(); train,test=load_cifar(cfg.data_root); labels=np.asarray(train.targets)
    test_loader=DataLoader(test,batch_size=256,shuffle=False,num_workers=cfg.num_workers,pin_memory=device.type=="cuda")
    if cfg.resume and cp.exists():
        saved=torch.load(cp,map_location="cpu",weights_only=False)
        if saved["experiment_id"]!=config_hash(cfg): raise ValueError("checkpoint configuration hash mismatch")
        start=saved["next_round"]; states=saved["client_states"]; splits=saved["data_split"]; graph=nx.node_link_graph(saved["graph"])
        records=saved["metrics"]; diss=[DissDLState.restore(x) for x in saved.get("baseline_state",[])]; restore_rng(saved["rng"])
        pstats=saved["partition_stats"]; initial=saved["initial_graph_stats"]
    else:
        splits=dirichlet_split(labels,cfg.num_clients,cfg.alpha,cfg.min_samples_per_client,cfg.seed,cfg.protocol=="canonical")
        pstats=partition_stats(labels,splits); states=initial_states(cfg.num_clients,cfg.seed); graph=initial_graph(cfg); start=0; records=[]
        initial=graph_stats(graph); initial["over_cap_nodes"]=sum(d>cfg.dmax for _,d in graph.degree())
        diss=[]
        if cfg.method=="dissdl":
            for i in range(cfg.num_clients):
                peers=set(range(cfg.num_clients))-{i}; diss.append(DissDLState(set(graph.neighbors(i)),peers))
            directed=nx.DiGraph(); directed.add_nodes_from(range(cfg.num_clients))
            for i,node in enumerate(diss): directed.add_edges_from((j,i) for j in node.wanted_senders)
            graph=directed
        write_edges(out/"graph_initial.edgelist",graph); atomic_json(out/"config.json",{**asdict(cfg),"experiment_id":config_hash(cfg),"deployment_protocol":"partial_participation" if cfg.participation_rate<1 else "full_participation"})
    fixed_eval=random.Random(cfg.seed+991).sample(range(cfg.num_clients),min(cfg.eval_clients,cfg.num_clients))
    # The reusable execution model must not perturb training/dropout RNG state.
    preserved_rng=rng_state(); working=CNN(); restore_rng(preserved_rng)
    model_bytes=sum(v.numel()*v.element_size() for v in states[0].values()); cumulative=records[-1]["cumulative_bytes"] if records else 0
    for rnd in range(start,cfg.rounds):
        # Method-independent schedule is required for paired comparisons.
        sampler=random.Random(cfg.seed*1_000_003+rnd)
        rt=time.perf_counter(); active=list(range(cfg.num_clients)) if cfg.participation_rate==1 else sorted(sampler.sample(range(cfg.num_clients),max(1,round(cfg.num_clients*cfg.participation_rate))))
        t=time.perf_counter()
        for i in active: states[i]=train_client(working,states[i],train,splits[i],cfg,device,cfg.seed*1_000_003+rnd*cfg.num_clients+i)
        train_s=time.perf_counter()-t
        if STOP_SIGNAL is not None:
            payload=checkpoint_payload(cfg,rnd,states,graph,splits,records,diss,pstats,initial); atomic_checkpoint(cp,payload); return EXIT_REQUEUE
        t=time.perf_counter()
        if cfg.method=="fedavg": tx=fedavg(states,active)
        elif cfg.method=="epidemic": graph=build_epidemic_graph(cfg.num_clients,cfg.dmax,cfg.seed+rnd); tx=epidemic_aggregate(states,graph,active)
        elif cfg.method=="dissdl": tx=dissdl_aggregate(states,diss,active)
        else: tx=aggregate(states,graph,active)
        agg_s=time.perf_counter()-t; topo_s=0.; trace=[]
        if rnd%cfg.topology_interval==0:
            t=time.perf_counter()
            if cfg.method=="lfhe": graph=lfhe_update(graph,[Adapter(s) for s in states],cfg.epsilon,cfg.dmax,cfg.w1,cfg.w2,cfg.w3,rnd,trace,None if cfg.update_mode=="sequential" and cfg.participation_rate==1 else active)
            elif cfg.method=="random_fof": graph,trace=fof_update(graph,active,cfg.dmax,True)
            elif cfg.method=="dissdl": graph=dissdl_update(diss,states,active)
            topo_s=time.perf_counter()-t
        control=sum(e.get("candidate_checks",0) for e in trace); cumulative += tx*model_bytes + control*16
        rec={"round":rnd,"active_clients":len(active),"model_transmissions":tx,"active_links":graph.number_of_edges(),"model_bytes":tx*model_bytes,
             "topology_control_messages":control,"topology_control_bytes":control*16,"cumulative_bytes":cumulative,"local_training_seconds":train_s,
             "aggregation_seconds":agg_s,"topology_update_seconds":topo_s,"candidate_checks":control,
             "fitness_evaluations":sum(e.get("fitness_evaluations",0) for e in trace),"accepted_additions":sum(e.get("action")=="accepted_addition" for e in trace),
             "accepted_swaps":sum(e.get("action")=="accepted_swap" for e in trace),"rejected_proposals":sum(e.get("action","").startswith("rejected") for e in trace)}
        if rnd%cfg.graph_metric_interval==0 or rnd==cfg.rounds-1: rec["graph"]=graph_stats(graph,True)
        if rnd%cfg.eval_interval==0 or rnd==cfg.rounds-1:
            et=time.perf_counter(); ids=list(range(cfg.num_clients)) if (cfg.protocol=="canonical" or (cfg.final_eval_all and rnd==cfg.rounds-1)) else fixed_eval
            acc,loss=evaluate(working,states,ids,test_loader,device); rec.update({"mean_accuracy":float(np.mean(acc)),"std_accuracy":float(np.std(acc)),"min_accuracy":min(acc),"max_accuracy":max(acc),"mean_loss":float(np.mean(loss)),"evaluated_clients":len(ids),"evaluation_seconds":time.perf_counter()-et}); rec.update(model_metrics(states))
        rec["total_round_seconds"]=time.perf_counter()-rt; rec["elapsed_seconds"]=time.time()-started
        try:
            import psutil; rec["peak_cpu_rss_bytes"]=psutil.Process().memory_info().rss
        except ImportError: rec["peak_cpu_rss_bytes"]=None
        rec["peak_gpu_allocated_bytes"]=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0; rec["peak_gpu_reserved_bytes"]=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
        records.append(rec); append_jsonl(out/"metrics.jsonl",rec)
        if (rnd+1)%cfg.checkpoint_interval==0 or STOP_SIGNAL is not None or rnd==cfg.rounds-1:
            ct=time.perf_counter(); atomic_checkpoint(cp,checkpoint_payload(cfg,rnd+1,states,graph,splits,records,diss,pstats,initial)); rec["checkpoint_seconds"]=time.perf_counter()-ct
        if STOP_SIGNAL is not None: return EXIT_REQUEUE
    final=graph_stats(graph); write_edges(out/"graph_final.edgelist",graph); atomic_json(out/"summary.json",summarize(cfg,records,started,pstats,initial,final,"partial_participation" if cfg.participation_rate<1 else "full_participation")); success.write_text("SUCCESS\n",encoding="utf-8"); return 0

def checkpoint_payload(cfg,next_round,states,graph,splits,records,diss,pstats,initial):
    return {"format_version":2,"next_round":next_round,"client_states":states,"optimizer_states":None,"graph_type":type(graph).__name__,"graph":nx.node_link_data(graph),
      "data_split":splits,"metrics":records,"rng":rng_state(),"configuration":asdict(cfg),"experiment_id":config_hash(cfg),"active_client_sampler_state":random.getstate(),
      "lfhe_state":{"update_mode":cfg.update_mode},"baseline_state":[x.checkpoint() for x in diss],"partition_stats":pstats,"initial_graph_stats":initial}

def parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--method",choices=METHODS,required=True); p.add_argument("--num-clients",type=int,required=True); p.add_argument("--seed",type=int,required=True)
    p.add_argument("--rounds",type=int); p.add_argument("--protocol",choices=("canonical","scalable"),required=True); p.add_argument("--alpha",type=float,default=.1); p.add_argument("--dmax",default="4",choices=("2","4","8","log2")); p.add_argument("--topology-interval",type=int); p.add_argument("--eval-interval",type=int)
    p.add_argument("--initial-graph",choices=("canonical_er","bounded_connected")); p.add_argument("--participation-rate",type=float,default=1.); group=p.add_mutually_exclusive_group(); group.add_argument("--local-epochs",type=int); group.add_argument("--local-steps",type=int)
    p.add_argument("--batch-size",type=int); p.add_argument("--lr",type=float,default=.05); p.add_argument("--output-dir",required=True); p.add_argument("--checkpoint-interval",type=int,default=10); p.add_argument("--checkpoint-path",default=""); p.add_argument("--resume",action="store_true"); p.add_argument("--force",action="store_true")
    p.add_argument("--eval-clients",type=int,default=50); p.add_argument("--final-eval-all",action=argparse.BooleanOptionalAction,default=True); p.add_argument("--data-root",default=os.getenv("LFHE_DATA_ROOT","./data")); p.add_argument("--update-mode",choices=("sequential","batched"),default="sequential")
    p.add_argument("--min-samples-per-client",type=int); p.add_argument("--target-accuracy",type=float,default=.65); p.add_argument("--graph-metric-interval",type=int,default=25); p.add_argument("--dissdl-max-n",type=int,default=500); p.add_argument("--num-workers",type=int,default=0); return p

def make_config(a):
    canonical=a.protocol=="canonical"; rounds=a.rounds if a.rounds is not None else 300; topo=a.topology_interval if a.topology_interval is not None else 5; ev=a.eval_interval if a.eval_interval is not None else 5
    initial=a.initial_graph or ("canonical_er" if canonical else "bounded_connected"); epochs=a.local_epochs; steps=a.local_steps
    if epochs is None and steps is None: epochs=1
    batch=a.batch_size or 32; minimum=a.min_samples_per_client if a.min_samples_per_client is not None else (1 if canonical else max(1,min(10,50000//a.num_clients//2)))
    dmax=math.ceil(math.log2(a.num_clients)) if a.dmax=="log2" else int(a.dmax)
    if not 0<a.participation_rate<=1: raise ValueError("participation-rate must be in (0,1]")
    if canonical and a.participation_rate!=1: raise ValueError("canonical protocol requires full participation")
    if canonical and (a.dmax!="4" or initial!="canonical_er" or topo!=5 or ev!=5 or batch!=32 or epochs!=1 or steps is not None): raise ValueError("canonical protocol requires D_max=4, canonical_er, intervals=5, batch=32, and one local epoch")
    if a.method=="dissdl" and a.num_clients>a.dissdl_max_n: raise ValueError("DissDL disabled above --dissdl-max-n due to its all-client known-peer directory")
    if a.method=="fedavg" and initial=="bounded_connected": initial="canonical_er"
    return Config(a.method,a.num_clients,a.seed,rounds,a.protocol,a.alpha,a.dmax,dmax,topo,ev,initial,a.participation_rate,epochs,steps,batch,a.lr,a.output_dir,a.checkpoint_interval,a.checkpoint_path,a.resume,a.force,a.eval_clients,a.final_eval_all,a.data_root,a.update_mode,minimum,a.target_accuracy,a.graph_metric_interval,a.dissdl_max_n,a.num_workers)

def main():
    try: return run(make_config(parser().parse_args()))
    except (ValueError,RuntimeError) as exc: print(f"error: {exc}",file=sys.stderr); return 2
if __name__=="__main__": raise SystemExit(main())
