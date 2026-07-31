#!/usr/bin/env python3
"""Canonical-compatible LFHE runner with resumable all-suite execution."""
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

try:
    from morph import MorphNode
    MORPH_IMPORT_ERROR = None
except ImportError as exc:
    MorphNode = None
    MORPH_IMPORT_ERROR = exc

EXIT_REQUEUE = 99
STOP_SIGNAL = None
METHODS = ("ring", "static_random", "epidemic", "dissdl", "random_fof", "morph", "lfhe", "fedavg")

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
    samples_per_client:int|None; link_failure_rate:float; stale_view_rounds:int
    repair_warning_fraction:float; representation_mode:str; lfhe_start_round:int
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
    return (datasets.CIFAR10(root,train=True,download=False,transform=train_tf),
            datasets.CIFAR10(root,train=False,download=False,transform=test_tf))

def dirichlet_split(labels, n, alpha, minimum, seed, preserve_original=False, max_attempts=1000,
                    return_stats=False, samples_per_client=None):
    """Original classwise split, deterministically resampled until the minimum holds."""
    if minimum*n > len(labels): raise ValueError("min-samples-per-client exceeds fixed dataset size")
    attempts=max_attempts if preserve_original else 1
    for attempt in range(attempts):
        rng=np.random.RandomState(seed+attempt); result=[[] for _ in range(n)]
        for c in range(int(labels.max())+1):
            idx=np.where(labels==c)[0]; rng.shuffle(idx)
            cuts=(np.cumsum(rng.dirichlet(np.repeat(alpha,n)))*len(idx)).astype(int)[:-1]
            for i,part in enumerate(np.split(idx,cuts)): result[i].extend(part.tolist())
        if min(map(len,result)) >= minimum: break
        if preserve_original and minimum == 0: break
    # Stable repair is the scalable fallback when rejection sampling is improbable.
    repaired=0
    while min(map(len,result)) < minimum:
        receiver=min(range(n),key=lambda i:(len(result[i]),i)); donor=max(range(n),key=lambda i:(len(result[i]),-i))
        result[receiver].append(result[donor].pop()); repaired+=1
    if samples_per_client is not None:
        if samples_per_client*n > len(labels): raise ValueError("samples-per-client exceeds CIFAR-10 capacity")
        if min(map(len,result)) < samples_per_client:
            while min(map(len,result)) < samples_per_client:
                receiver=min(range(n),key=lambda i:(len(result[i]),i)); donor=max(range(n),key=lambda i:(len(result[i]),-i))
                if len(result[donor])<=samples_per_client: raise RuntimeError("cannot construct fixed-samples-per-client split")
                result[receiver].append(result[donor].pop()); repaired+=1
        result=[indices[:samples_per_client] for indices in result]
    meta={"partition_resampling_attempts":attempt+1,"repaired_samples":repaired,
          "repaired_sample_fraction":repaired/max(1,sum(map(len,result)))}
    return (result,meta) if return_stats else result

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

def clustered_hard(n,dmax,seed,connected=True):
    """Two dense-ish ring communities; one bridge when connected."""
    if n<6 or dmax<2: raise ValueError("clustered topology requires N>=6 and D_max>=2")
    cut=n//2; g=nx.disjoint_union(nx.cycle_graph(cut),nx.cycle_graph(n-cut))
    if connected:
        g.remove_edge(0,cut-1); g.remove_edge(cut,n-1); g.add_edge(0,cut); g.add_edge(cut-1,n-1)
    if dmax>2:
        rng=random.Random(seed); groups=[list(range(cut)),list(range(cut,n))]
        for group in groups:
            pairs=[(u,v) for u in group for v in group if u<v]; rng.shuffle(pairs)
            for u,v in pairs:
                if g.degree(u)<dmax and g.degree(v)<dmax and not g.has_edge(u,v): g.add_edge(u,v)
    return g

def initial_graph(cfg):
    if cfg.method=="morph":
        if cfg.dmax>=cfg.num_clients: raise ValueError("Morph in-degree must be smaller than num-clients")
        rng=random.Random(cfg.seed); graph=nx.DiGraph(); graph.add_nodes_from(range(cfg.num_clients))
        for receiver in range(cfg.num_clients):
            peers=[i for i in range(cfg.num_clients) if i!=receiver]
            graph.add_edges_from((sender,receiver) for sender in rng.sample(peers,cfg.dmax))
        return graph
    if cfg.method=="ring": return nx.cycle_graph(cfg.num_clients)
    if cfg.method=="epidemic" and cfg.protocol=="canonical": return build_epidemic_graph(cfg.num_clients,cfg.dmax,cfg.seed)
    if cfg.method=="dissdl" and cfg.protocol=="canonical":
        rng=random.Random(cfg.seed); graph=nx.DiGraph(); graph.add_nodes_from(range(cfg.num_clients))
        for i in range(cfg.num_clients):
            peers=[j for j in range(cfg.num_clients) if j!=i]
            graph.add_edges_from((j,i) for j in rng.sample(peers,min(3,len(peers))))
        return graph
    if cfg.initial_graph=="canonical_er": return canonical_er(cfg.num_clients,cfg.seed)
    if cfg.initial_graph=="clustered": return clustered_hard(cfg.num_clients,cfg.dmax,cfg.seed,True)
    if cfg.initial_graph=="disconnected_clusters": return clustered_hard(cfg.num_clients,cfg.dmax,cfg.seed,False)
    return bounded_connected(cfg.num_clients,cfg.dmax,cfg.seed)

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

def failed_link_view(graph, rate, seed):
    if rate<=0: return graph,0
    rng=random.Random(seed); view=graph.copy(); removed=[]
    for edge in list(view.edges()):
        if rng.random()<rate: removed.append(edge)
    view.remove_edges_from(removed)
    return view,len(removed)

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
    def __init__(self,state,mode="flatten"):
        self.weight=state["classifier.4.weight"] if isinstance(state,dict) else state
        self.mode=mode
    def get_representation(self):
        if self.mode=="class_mean":
            return self.weight.mean(dim=1).flatten()
        return self.weight.flatten()
class Adapter:
    """Lightweight LFHE/Morph interface; does not allocate another CNN."""
    def __init__(self,state,mode="flatten"): self.model=_RepresentationModel(state,mode)

def make_morph_nodes(states, graph, cfg):
    nodes=[]
    for i,state in enumerate(states):
        model=CNN(); model.load_state_dict(state)
        nodes.append(MorphNode(i,model,list(graph.predecessors(i)),in_degree=cfg.dmax,
            beta=500.0,change_iter=cfg.topology_interval,seed=cfg.seed,
            indirect_history_k=5,device=torch.device("cpu")))
    return nodes

def morph_round(states, nodes, active, round_index):
    """Execute the MorphNode protocol used by morph_matched_rerun.py."""
    active=set(active)
    for i in active:
        nodes[i].model.load_state_dict(states[i],strict=True)
        nodes[i].begin_round(round_index)
        nodes[i].update_wanted_senders(round_index,available_peers=active)
        nodes[i].validate_state()
    payloads={i:nodes[i].build_model_payload(degree=len(nodes[i].wanted_senders)) for i in active}
    transmissions=0
    for receiver in active:
        node=nodes[receiver]
        for sender in node.requested_senders():
            if sender in active and nodes[sender].should_send_to(receiver,True):
                node.receive_model_payload(sender,payloads[sender]); transmissions+=1
    for i in active:
        nodes[i].aggregate(); states[i]=clone_state(nodes[i].model.state_dict())
    graph=nx.DiGraph(); graph.add_nodes_from(range(len(nodes)))
    for receiver,node in enumerate(nodes):
        graph.add_edges_from((sender,receiver) for sender in node.wanted_senders)
    return graph,transmissions

def morph_checkpoint(nodes):
    return [{"wanted_senders":sorted(n.wanted_senders),"known_nodes":sorted(n.known_nodes),
      "peer_models":n.peer_models,"has_real_model":sorted(n.has_real_model),
      "similarity_cache":n.similarity_cache,
      "sim_estimates":{peer:list(values) for peer,values in n.sim_estimates_per_peer.items()},
      "iteration":n.iteration,"last_added_peer":n.last_added_peer,"last_removed_peer":n.last_removed_peer,
      "topology_change_count":n.topology_change_count,"rng_state":n.rng.getstate()} for n in nodes]

def restore_morph_nodes(states, graph, cfg, saved):
    nodes=make_morph_nodes(states,graph,cfg)
    for node,value in zip(nodes,saved):
        node.wanted_senders=set(value["wanted_senders"]); node.known_nodes=set(value["known_nodes"])
        node.peer_models=value["peer_models"]; node.has_real_model=set(value["has_real_model"])
        node.similarity_cache=value["similarity_cache"]; node.sim_estimates_per_peer.clear()
        for peer,estimates in value["sim_estimates"].items(): node.sim_estimates_per_peer[int(peer)].extend(estimates)
        node.iteration=value["iteration"]; node.last_added_peer=value["last_added_peer"]
        node.last_removed_peer=value["last_removed_peer"]; node.topology_change_count=value["topology_change_count"]
        node.rng.setstate(value["rng_state"])
    return nodes

def fof_update(g, active, dmax, random_policy=False):
    g=g.copy(); trace=[]
    for i in active:
        event={"client":int(i),"candidate_checks":0,"fitness_evaluations":0,"action":"no_candidate"}; ni=list(g.neighbors(i))
        if ni:
            j=random.choice(ni); candidates=[k for k in g.neighbors(j) if k!=i and not g.has_edge(i,k)]
            if candidates:
                k=random.choice(candidates); event["candidate"]=int(k); event["candidate_checks"]=1; event["action"]="rejected_proposal"
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

def snapshot_concurrent_lfhe(graph, clients, active, cfg, round_index):
    """Propose on one immutable snapshot, then commit conflict-free proposals.

    Proposal generation calls canonical single-client LFHE semantics. Commit order is
    deterministic by client id and rejects shared endpoints or stale infeasibility.
    """
    snapshot=graph.copy(); proposals=[]; trace=[]
    def edges(g): return {tuple(sorted((int(u),int(v)))) for u,v in g.edges()}
    for i in sorted(active):
        local_trace=[]
        proposed=lfhe_update(snapshot,clients,cfg.epsilon,cfg.dmax,cfg.w1,cfg.w2,cfg.w3,
                             round_index,local_trace,[i])
        event=local_trace[0] if local_trace else {"client":i,"action":"no_candidate","candidate_checks":0,"fitness_evaluations":0}
        added=edges(proposed)-edges(snapshot); removed=edges(snapshot)-edges(proposed)
        if added or removed: proposals.append((i,added,removed,event))
        trace.append(event)
    committed=snapshot.copy(); touched=set(); conflicts=stale_rejections=degree_rejections=0
    for i,added,removed,event in proposals:
        endpoints={x for edge in added|removed for x in edge}
        if endpoints&touched:
            conflicts+=1; event["action"]="rejected_shared_endpoint_conflict"; continue
        candidate=committed.copy()
        if any(not candidate.has_edge(*edge) for edge in removed):
            stale_rejections+=1; event["action"]="rejected_stale_topology"; continue
        candidate.remove_edges_from(removed); candidate.add_edges_from(added)
        if max(dict(candidate.degree()).values())>cfg.dmax:
            degree_rejections+=1; event["action"]="rejected_degree_safety"; continue
        committed=candidate; touched.update(endpoints); event["action"]="committed_"+event["action"]
    stats={"proposal_count":len(proposals),"shared_endpoint_conflicts":conflicts,
           "shared_endpoint_conflict_rate":conflicts/max(1,len(proposals)),
           "stale_rejections":stale_rejections,"degree_safety_rejections":degree_rejections,
           "committed_proposals":len(proposals)-conflicts-stale_rejections-degree_rejections}
    return committed,trace,stats

def dissdl_aggregate(states, nodes, active):
    active=set(active); updates={}; transmissions=0
    for i in active:
        incoming=[j for j in nodes[i].wanted_senders if j in active]; transmissions+=len(incoming)
        updates[i]=weighted_average([(1/(len(incoming)+1),states[j]) for j in incoming]+[(1/(len(incoming)+1),states[i])])
        def parameters(state): return torch.cat([value.flatten().float() for key,value in state.items() if not any(token in key for token in ("running_mean","running_var","num_batches_tracked"))])
        rep_i=parameters(states[i])
        for j in incoming:
            rep_j=parameters(states[j]); sim=float(torch.nn.functional.cosine_similarity(rep_i,rep_j,dim=0))
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

def summarize(cfg, records, started, pstats, initial, final, deployment, checkpoint_path=None):
    evals=[r for r in records if "mean_accuracy" in r]; xs=[r["round"] for r in evals]; ys=[r["mean_accuracy"] for r in evals]
    integrate=getattr(np,"trapezoid",getattr(np,"trapz",None))
    auc=float(integrate(ys,xs)/(xs[-1]-xs[0])) if len(xs)>1 else (ys[0] if ys else None)
    reached=next((r for r in evals if r["mean_accuracy"]>=cfg.target_accuracy),None)
    accepted=[r["round"] for r in records if r.get("accepted_additions",0)+r.get("accepted_swaps",0)>0]
    projection_records=records[:-1] if len(records)>1 else records
    mean_round=float(np.mean([r["total_round_seconds"] for r in projection_records])) if projection_records else None
    peak_rss=max((r.get("peak_cpu_rss_bytes") or 0 for r in records),default=0); peak_gpu=max((r.get("peak_gpu_reserved_bytes") or 0 for r in records),default=0)
    checkpoint_bytes=checkpoint_path.stat().st_size if checkpoint_path and checkpoint_path.exists() else None
    checkpoint_seconds=max((r.get("checkpoint_seconds",0) for r in records),default=0); final_eval=evals[-1].get("evaluation_seconds") if evals else None
    projected=mean_round*300 if mean_round is not None else None
    gpu_total=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
    gate_checks={"projected_300_round_seconds_le_48h":projected is not None and projected<=48*3600,
      "peak_cpu_rss_le_22_4_gib":peak_rss<=22.4*1024**3,"peak_gpu_reserved_le_90_percent":gpu_total is None or peak_gpu<=.9*gpu_total,
      "checkpoint_le_10_gib":checkpoint_bytes is not None and checkpoint_bytes<=10*1024**3,"checkpoint_time_le_600s":checkpoint_seconds<=600,
      "full_evaluation_time_le_3h":final_eval is not None and final_eval<=3*3600,"repair_fraction_le_threshold":pstats.get("repaired_sample_fraction",0)<=cfg.repair_warning_fraction,
      "no_nan_or_inf":all(r.get("model_parameters_finite",True) and all(not isinstance(v,float) or math.isfinite(v) for v in r.values()) for r in records)}
    return {"experiment_id":config_hash(cfg),"status":"complete","protocol":cfg.protocol,"deployment_protocol":deployment,
      "final_accuracy":ys[-1] if ys else None,"normalized_auc":auc,"rounds_to_target":reached["round"] if reached else None,
      "bytes_to_target":reached["cumulative_bytes"] if reached else None,"wall_clock_time_to_target":reached["elapsed_seconds"] if reached else None,
      "partition":pstats,"initial_graph":initial,"final_graph":final,"wall_clock_seconds":time.time()-started,
      "evaluations":len(evals),"rounds_completed":cfg.rounds,"last_accepted_rewire_round":accepted[-1] if accepted else None,
      "topology_stabilization_round":(accepted[-1]+1) if accepted else 0,
      "resource_projection":{"mean_round_seconds":mean_round,"projected_300_round_seconds":projected,"peak_cpu_rss_bytes":peak_rss,"peak_gpu_reserved_bytes":peak_gpu,"checkpoint_bytes":checkpoint_bytes,"max_checkpoint_seconds":checkpoint_seconds,"final_full_evaluation_seconds":final_eval},
      "feasibility_gate":{"passed":all(gate_checks.values()),"checks":gate_checks},
      "scientific_notes":["canonical LFHE fitness/annealing/representation" if cfg.method=="lfhe" else "real Morph topology implementation" if cfg.method=="morph" else "matched baseline budget"]}

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
        rep_history=saved.get("lfhe_state",{}).get("representation_history",[])
        pstats=saved["partition_stats"]; initial=saved["initial_graph_stats"]
        morph_nodes=restore_morph_nodes(states,graph,cfg,saved.get("morph_state",[])) if cfg.method=="morph" else []
    else:
        samples=cfg.samples_per_client if cfg.data_regime=="fixed_per_client" else None
        splits,split_meta=dirichlet_split(labels,cfg.num_clients,cfg.alpha,cfg.min_samples_per_client,cfg.seed,cfg.protocol=="canonical",return_stats=True,samples_per_client=samples)
        pstats={**partition_stats(labels,splits),**split_meta}; pstats["repair_warning"]=pstats["repaired_sample_fraction"]>cfg.repair_warning_fraction
        states=initial_states(cfg.num_clients,cfg.seed); graph=initial_graph(cfg); start=0; records=[]; rep_history=[]
        diss=[]; morph_nodes=[]
        if cfg.method=="dissdl":
            for i in range(cfg.num_clients):
                peers=set(range(cfg.num_clients))-{i}; initial_senders=set(graph.predecessors(i)) if graph.is_directed() else set(graph.neighbors(i)); diss.append(DissDLState(initial_senders,peers))
            if not graph.is_directed():
                directed=nx.DiGraph(); directed.add_nodes_from(range(cfg.num_clients))
                for i,node in enumerate(diss): directed.add_edges_from((j,i) for j in node.wanted_senders)
                graph=directed
        if cfg.method=="morph": morph_nodes=make_morph_nodes(states,graph,cfg)
        initial=graph_stats(graph); initial["over_cap_nodes"]=sum(d>cfg.dmax for _,d in graph.degree())
        representation_shape=list(states[0]["classifier.4.weight"].shape)
        representation_dimension=(representation_shape[0] if cfg.representation_mode=="class_mean" else int(states[0]["classifier.4.weight"].numel()))
        resolved_initial="dissdl_random_in_degree_3" if cfg.protocol=="canonical" and cfg.method=="dissdl" else "epidemic_directed_degree_4" if cfg.protocol=="canonical" and cfg.method=="epidemic" else cfg.initial_graph
        write_edges(out/"graph_initial.edgelist",graph); atomic_json(out/"config.json",{**asdict(cfg),"resolved_initial_graph":resolved_initial,"experiment_id":config_hash(cfg),"deployment_protocol":"partial_participation" if cfg.participation_rate<1 else "full_participation","representation_shape":representation_shape,"representation_dimension":representation_dimension,"representation_mode":cfg.representation_mode,"lfhe_start_round":cfg.lfhe_start_round})
        print(f"[representation] shape={tuple(states[0]['classifier.4.weight'].shape)} flattened_dimension={representation_dimension}",flush=True)
    fixed_eval=random.Random(cfg.seed+991).sample(range(cfg.num_clients),min(cfg.eval_clients,cfg.num_clients))
    # The reusable execution model must not perturb training/dropout RNG state.
    preserved_rng=rng_state(); working=CNN(); restore_rng(preserved_rng)
    model_bytes=sum(v.numel()*v.element_size() for v in states[0].values()); cumulative=records[-1]["cumulative_bytes"] if records else 0
    previous_effective_connected=records[-1].get("effective_connected",True) if records else True
    disconnect_streak=0
    for previous in reversed(records):
        if previous.get("effective_connected",True): break
        disconnect_streak+=1
    for rnd in range(start,cfg.rounds):
        # Method-independent schedule is required for paired comparisons.
        sampler=random.Random(cfg.seed*1_000_003+rnd)
        rt=time.perf_counter(); active=list(range(cfg.num_clients)) if cfg.participation_rate==1 else sorted(sampler.sample(range(cfg.num_clients),max(1,round(cfg.num_clients*cfg.participation_rate))))
        t=time.perf_counter()
        for i in active: states[i]=train_client(working,states[i],train,splits[i],cfg,device,cfg.seed*1_000_003+rnd*cfg.num_clients+i)
        train_s=time.perf_counter()-t
        if STOP_SIGNAL is not None:
            payload=checkpoint_payload(cfg,rnd,states,graph,splits,records,diss,pstats,initial,rep_history,morph_nodes); atomic_checkpoint(cp,payload); return EXIT_REQUEUE
        t=time.perf_counter()
        aggregation_graph,dropped_links=failed_link_view(graph,cfg.link_failure_rate,cfg.seed*10_000_019+rnd)
        if cfg.method=="fedavg": tx=fedavg(states,active)
        elif cfg.method=="epidemic": graph=build_epidemic_graph(cfg.num_clients,cfg.dmax,cfg.seed+rnd); aggregation_graph,dropped_links=failed_link_view(graph,cfg.link_failure_rate,cfg.seed*10_000_019+rnd); tx=epidemic_aggregate(states,aggregation_graph,active)
        elif cfg.method=="dissdl": tx=dissdl_aggregate(states,diss,active)
        elif cfg.method=="morph": graph,tx=morph_round(states,morph_nodes,active,rnd); aggregation_graph=graph
        else: tx=aggregate(states,aggregation_graph,active)
        agg_s=time.perf_counter()-t; topo_s=0.; trace=[]; concurrency={"proposal_count":0,"shared_endpoint_conflicts":0,"shared_endpoint_conflict_rate":0.,"stale_rejections":0,"degree_safety_rejections":0,"committed_proposals":0}
        current_reps=torch.stack([s["classifier.4.weight"].clone() for s in states]); rep_history.append((rnd,current_reps)); rep_history=rep_history[-max(1,cfg.stale_view_rounds+1):]
        lfhe_window=(cfg.method!="lfhe" or rnd>=cfg.lfhe_start_round)
        if cfg.method!="morph" and rnd%cfg.topology_interval==0 and lfhe_window:
            t=time.perf_counter()
            component_id={node:index for index,component in enumerate(nx.connected_components(graph.to_undirected())) for node in component}
            view_index=max(0,len(rep_history)-1-cfg.stale_view_rounds); view_round,view_reps=rep_history[view_index]; clients=[Adapter(view_reps[i],cfg.representation_mode) for i in range(cfg.num_clients)]
            if cfg.method=="lfhe" and cfg.update_mode=="snapshot_concurrent": graph,trace,concurrency=snapshot_concurrent_lfhe(graph,clients,active,cfg,rnd)
            elif cfg.method=="lfhe": graph=lfhe_update(graph,clients,cfg.epsilon,cfg.dmax,cfg.w1,cfg.w2,cfg.w3,rnd,trace,None if cfg.update_mode=="sequential" and cfg.participation_rate==1 else active)
            elif cfg.method=="random_fof": graph,trace=fof_update(graph,active,cfg.dmax,True)
            elif cfg.method=="dissdl": graph=dissdl_update(diss,states,active)
            topo_s=time.perf_counter()-t
        else: component_id={node:index for index,component in enumerate(nx.connected_components(graph.to_undirected())) for node in component}
        control=sum(e.get("candidate_checks",0) for e in trace); cumulative += tx*model_bytes + control*16
        effective_components=nx.number_connected_components(aggregation_graph.to_undirected()); effective_connected=effective_components==1
        prior_disconnect_streak=disconnect_streak; disconnect_streak=0 if effective_connected else disconnect_streak+1
        recovery_rounds=prior_disconnect_streak if effective_connected and not previous_effective_connected else None
        rec={"round":rnd,"active_clients":len(active),"model_transmissions":tx,"active_links":graph.number_of_edges(),"model_bytes":tx*model_bytes,
             "topology_control_messages":control,"topology_control_bytes":control*16,"cumulative_bytes":cumulative,"local_training_seconds":train_s,
             "aggregation_seconds":agg_s,"topology_update_seconds":topo_s,"candidate_checks":control,
             "link_failure_rate":cfg.link_failure_rate,"failed_links":dropped_links,"effective_connected_components":effective_components,"effective_connected":effective_connected,"recovered_this_round":effective_connected and not previous_effective_connected,"recovery_rounds":recovery_rounds,
             "stale_view_rounds":cfg.stale_view_rounds,"representation_view_round":view_round if rnd%cfg.topology_interval==0 and lfhe_window else None,**concurrency,
             "fitness_evaluations":sum(e.get("fitness_evaluations",0) for e in trace),"accepted_additions":sum(e.get("action","").endswith("accepted_addition") for e in trace),
             "accepted_swaps":sum(e.get("action","").endswith("accepted_swap") for e in trace),"rejected_proposals":sum(e.get("action","").startswith("rejected") for e in trace),
             "fof_cross_component_proposals":sum("candidate" in e and component_id.get(e.get("client"))!=component_id.get(e.get("candidate")) for e in trace),"fof_component_count":len(set(component_id.values()))}
        if rnd%cfg.graph_metric_interval==0 or rnd==cfg.rounds-1: rec["graph"]=graph_stats(graph,True)
        if rnd%cfg.eval_interval==0 or rnd==cfg.rounds-1:
            et=time.perf_counter(); ids=list(range(cfg.num_clients)) if (cfg.protocol=="canonical" or (cfg.final_eval_all and rnd==cfg.rounds-1)) else fixed_eval
            acc,loss=evaluate(working,states,ids,test_loader,device); rec.update({"mean_accuracy":float(np.mean(acc)),"std_accuracy":float(np.std(acc)),"min_accuracy":min(acc),"max_accuracy":max(acc),"mean_loss":float(np.mean(loss)),"evaluated_clients":len(ids),"evaluation_seconds":time.perf_counter()-et}); rec.update(model_metrics(states)); rec["model_parameters_finite"]=all(torch.isfinite(v).all().item() for state in states for v in state.values() if v.is_floating_point())
        rec["total_round_seconds"]=time.perf_counter()-rt; rec["elapsed_seconds"]=time.time()-started
        try:
            import psutil; rec["peak_cpu_rss_bytes"]=psutil.Process().memory_info().rss
        except ImportError: rec["peak_cpu_rss_bytes"]=None
        rec["peak_gpu_allocated_bytes"]=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0; rec["peak_gpu_reserved_bytes"]=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
        records.append(rec)
        if (rnd+1)%cfg.checkpoint_interval==0 or STOP_SIGNAL is not None or rnd==cfg.rounds-1:
            ct=time.perf_counter(); atomic_checkpoint(cp,checkpoint_payload(cfg,rnd+1,states,graph,splits,records,diss,pstats,initial,rep_history,morph_nodes)); rec["checkpoint_seconds"]=time.perf_counter()-ct
        append_jsonl(out/"metrics.jsonl",rec)
        if STOP_SIGNAL is not None: return EXIT_REQUEUE
        previous_effective_connected=effective_connected
    final=graph_stats(graph); write_edges(out/"graph_final.edgelist",graph); atomic_json(out/"summary.json",summarize(cfg,records,started,pstats,initial,final,"partial_participation" if cfg.participation_rate<1 else "full_participation",cp)); success.write_text("SUCCESS\n",encoding="utf-8"); return 0

def checkpoint_payload(cfg,next_round,states,graph,splits,records,diss,pstats,initial,rep_history=None,morph_nodes=None):
    return {"format_version":2,"next_round":next_round,"client_states":states,"optimizer_states":None,"graph_type":type(graph).__name__,"graph":nx.node_link_data(graph),
      "data_split":splits,"metrics":records,"rng":rng_state(),"configuration":asdict(cfg),"experiment_id":config_hash(cfg),"active_client_sampler_state":random.getstate(),
      "lfhe_state":{"update_mode":cfg.update_mode,"representation_history":rep_history or []},"baseline_state":[x.checkpoint() for x in diss],
      "morph_state":morph_checkpoint(morph_nodes or []),"partition_stats":pstats,"initial_graph_stats":initial}



def experiment_complete(output_dir: str | Path) -> bool:
    """Return True only when both SUCCESS and a complete summary are present."""
    out=Path(output_dir); success=out/"SUCCESS"; summary=out/"summary.json"
    if not success.exists() or not summary.exists(): return False
    try:
        payload=json.loads(summary.read_text(encoding="utf-8"))
        return payload.get("status")=="complete"
    except (OSError,json.JSONDecodeError):
        return False


def parse_int_csv(value):
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def batch_parser():
    p=argparse.ArgumentParser(description="Run every LFHE workshop experiment from one command.")
    p.add_argument("--run-all",action="store_true",required=True)
    p.add_argument("--suite",choices=("all","paper","canonical","feasibility","primary","degree","partial"),default="all")
    p.add_argument("--results-root",default="./results/workshop_all")
    p.add_argument("--data-root",default=os.getenv("LFHE_DATA_ROOT","./data"))
    p.add_argument("--seeds",default="42,43,44",help="Seeds for scalable suites.")
    p.add_argument("--canonical-seeds",default="42,43,44,45,46")
    p.add_argument("--primary-clients",default="100,200,500,1000,1500,2000")
    p.add_argument("--degree-clients",default="100,500,1000,2000")
    p.add_argument("--partial-clients",default="1000,1500,2000")
    p.add_argument("--force",action="store_true",help="Delete and rerun completed/incomplete outputs.")
    p.add_argument("--continue-on-error",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--dry-run",action="store_true")
    return p


def _spec(method,n,seed,protocol,rounds,stage,**kwargs):
    return {"method":method,"num_clients":n,"seed":seed,"protocol":protocol,
            "rounds":rounds,"stage":stage,**kwargs}


def build_workshop_specs(a):
    seeds=parse_int_csv(a.seeds); canonical_seeds=parse_int_csv(a.canonical_seeds)
    primary_clients=parse_int_csv(a.primary_clients); degree_clients=parse_int_csv(a.degree_clients)
    partial_clients=parse_int_csv(a.partial_clients); specs=[]
    selected={a.suite} if a.suite!="all" else {"paper","canonical","feasibility","primary","degree","partial"}
    if "paper" in selected:
        # Exact main-paper scalability protocol, plus Morph.
        for n in (10,50,100,500):
            for method in ("ring","static_random","epidemic","dissdl","morph","lfhe"):
                for seed in canonical_seeds[:3]:
                    specs.append(_spec(method,n,seed,"scalable",300,"paper_alignment",
                        alpha=.3,dmax=str(max(2,int(2*math.log(n)))),initial_graph="canonical_er",
                        topology_interval=5,eval_interval=5,eval_clients=n,final_eval_all=True,
                        representation_mode="class_mean",lfhe_start_round=0))
    if "canonical" in selected:
        for method in ("ring","static_random","epidemic","dissdl","random_fof","morph","lfhe"):
            for seed in canonical_seeds: specs.append(_spec(method,30,seed,"canonical",300,"canonical"))
    if "feasibility" in selected:
        for n in (100,500,1000,1500,2000):
            for method in ("static_random","morph","lfhe"):
                specs.append(_spec(method,n,seeds[0],"scalable",5,"feasibility_full"))
        for n in partial_clients:
            for method in ("static_random","lfhe"):
                specs.append(_spec(method,n,seeds[0],"scalable",5,"feasibility_partial",participation_rate=.1))
    if "primary" in selected:
        for n in primary_clients:
            for method in ("static_random","random_fof","morph","lfhe"):
                for seed in seeds: specs.append(_spec(method,n,seed,"scalable",300,"primary"))
    if "degree" in selected:
        for n in degree_clients:
            for dmax in ("2","4","8","log2"):
                for method in ("static_random","random_fof","morph","lfhe"):
                    for seed in seeds: specs.append(_spec(method,n,seed,"scalable",300,"degree",dmax=dmax))
    if "partial" in selected:
        for n in partial_clients:
            for method in ("static_random","morph","lfhe"):
                for seed in seeds: specs.append(_spec(method,n,seed,"scalable",300,"partial",participation_rate=.1))
    return specs


def namespace_from_spec(spec,a):
    stage=spec["stage"]; dmax=spec.get("dmax","4"); participation=spec.get("participation_rate",1.)
    suffix=f"n{spec['num_clients']}_seed{spec['seed']}_d{dmax}_p{participation:g}"
    output=Path(a.results_root)/stage/spec["method"]/suffix
    # Construct the same argparse namespace expected by make_config.
    return argparse.Namespace(method=spec["method"],num_clients=spec["num_clients"],seed=spec["seed"],
      rounds=spec["rounds"],protocol=spec["protocol"],alpha=spec.get("alpha",.1),dmax=dmax,topology_interval=spec.get("topology_interval"),
      eval_interval=spec.get("eval_interval"),initial_graph=spec.get("initial_graph"),participation_rate=participation,local_epochs=spec.get("local_epochs"),
      local_steps=spec.get("local_steps"),batch_size=spec.get("batch_size"),lr=spec.get("lr",.05),output_dir=str(output),checkpoint_interval=spec.get("checkpoint_interval",10),
      checkpoint_path="",resume=(output/"checkpoint.pt").exists(),force=a.force,eval_clients=spec.get("eval_clients",50),
      final_eval_all=spec.get("final_eval_all",True),data_root=a.data_root,update_mode=spec.get("update_mode","sequential"),data_regime=spec.get("data_regime","fixed_total"),
      samples_per_client=spec.get("samples_per_client"),link_failure_rate=spec.get("link_failure_rate",0.),stale_view_rounds=spec.get("stale_view_rounds",0),repair_warning_fraction=.05,
      min_samples_per_client=spec.get("min_samples_per_client"),representation_mode=spec.get("representation_mode","flatten"),lfhe_start_round=spec.get("lfhe_start_round",0),target_accuracy=.65,graph_metric_interval=25,dissdl_max_n=500,
      num_workers=0)


def run_all(a):
    specs=build_workshop_specs(a); total=len(specs); skipped=completed=failed=0
    print(f"[suite] {a.suite}: {total} experiments",flush=True)
    for index,spec in enumerate(specs,1):
        ns=namespace_from_spec(spec,a); out=Path(ns.output_dir)
        label=f"[{index}/{total}] {spec['stage']} {spec['method']} N={spec['num_clients']} seed={spec['seed']}"
        if experiment_complete(out) and not a.force:
            print(f"[skip-complete] {label}",flush=True); skipped+=1; continue
        if (out/"checkpoint.pt").exists() and not a.force:
            ns.resume=True; print(f"[resume] {label}",flush=True)
        else: print(f"[run] {label}",flush=True)
        if a.dry_run: continue
        try:
            code=run(make_config(ns))
            if code==EXIT_REQUEUE:
                print(f"[stopped] checkpoint saved for {label}",flush=True); return code
            if code!=0: raise RuntimeError(f"experiment exited with code {code}")
            completed+=1
        except Exception as exc:
            failed+=1; print(f"[failed] {label}: {type(exc).__name__}: {exc}",file=sys.stderr,flush=True)
            if not a.continue_on_error: return 2
    print(f"[suite-done] newly_completed={completed} skipped={skipped} failed={failed}",flush=True)
    return 0 if failed==0 else 2

def parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--method",choices=METHODS,required=True); p.add_argument("--num-clients",type=int,required=True); p.add_argument("--seed",type=int,required=True)
    p.add_argument("--rounds",type=int); p.add_argument("--protocol",choices=("canonical","scalable"),required=True); p.add_argument("--alpha",type=float,default=.1); p.add_argument("--dmax",default="4",help="Positive integer degree cap or log2"); p.add_argument("--topology-interval",type=int); p.add_argument("--eval-interval",type=int)
    p.add_argument("--initial-graph",choices=("canonical_er","bounded_connected","clustered","disconnected_clusters")); p.add_argument("--participation-rate",type=float,default=1.); group=p.add_mutually_exclusive_group(); group.add_argument("--local-epochs",type=int); group.add_argument("--local-steps",type=int)
    p.add_argument("--batch-size",type=int); p.add_argument("--lr",type=float,default=.05); p.add_argument("--output-dir",required=True); p.add_argument("--checkpoint-interval",type=int,default=10); p.add_argument("--checkpoint-path",default=""); p.add_argument("--resume",action="store_true"); p.add_argument("--force",action="store_true")
    p.add_argument("--eval-clients",type=int,default=50); p.add_argument("--final-eval-all",dest="final_eval_all",action="store_true"); p.add_argument("--no-final-eval-all",dest="final_eval_all",action="store_false"); p.set_defaults(final_eval_all=True); p.add_argument("--data-root",default=os.getenv("LFHE_DATA_ROOT","./data")); p.add_argument("--update-mode",choices=("sequential","snapshot_concurrent"),default="sequential")
    p.add_argument("--data-regime",choices=("fixed_total","fixed_per_client"),default="fixed_total"); p.add_argument("--samples-per-client",type=int)
    p.add_argument("--link-failure-rate",type=float,default=0.); p.add_argument("--stale-view-rounds",type=int,default=0); p.add_argument("--repair-warning-fraction",type=float,default=.05)
    p.add_argument("--min-samples-per-client",type=int); p.add_argument("--representation-mode",choices=("flatten","class_mean"),default="flatten"); p.add_argument("--lfhe-start-round",type=int,default=0); p.add_argument("--target-accuracy",type=float,default=.65); p.add_argument("--graph-metric-interval",type=int,default=25); p.add_argument("--dissdl-max-n",type=int,default=500); p.add_argument("--num-workers",type=int,default=0); return p

def make_config(a):
    canonical=a.protocol=="canonical"; rounds=a.rounds if a.rounds is not None else 300; topo=a.topology_interval if a.topology_interval is not None else 5; ev=a.eval_interval if a.eval_interval is not None else 5
    initial=a.initial_graph or ("canonical_er" if canonical else "bounded_connected"); epochs=a.local_epochs; steps=a.local_steps
    if epochs is None and steps is None: epochs=1
    batch=a.batch_size or 32; minimum=a.min_samples_per_client if a.min_samples_per_client is not None else (1 if canonical else max(1,min(10,50000//a.num_clients//2)))
    dmax=math.ceil(math.log2(a.num_clients)) if a.dmax=="log2" else int(a.dmax)
    if dmax<2: raise ValueError("dmax must be >=2")
    if not 0<a.participation_rate<=1: raise ValueError("participation-rate must be in (0,1]")
    if not 0<=a.link_failure_rate<1: raise ValueError("link-failure-rate must be in [0,1)")
    if a.stale_view_rounds<0: raise ValueError("stale-view-rounds must be >=0")
    if a.lfhe_start_round<0: raise ValueError("lfhe-start-round must be >=0")
    if a.data_regime=="fixed_per_client" and (a.samples_per_client is None or a.samples_per_client<1): raise ValueError("fixed_per_client requires --samples-per-client")
    if a.samples_per_client is not None and a.samples_per_client*a.num_clients>50000: raise ValueError("requested fixed-per-client data exceeds CIFAR-10 training set")
    if canonical and a.participation_rate!=1: raise ValueError("canonical protocol requires full participation")
    if canonical:
        a.representation_mode="flatten"
        a.lfhe_start_round=21
    if canonical and (a.dmax!="4" or initial!="canonical_er" or topo!=5 or ev!=5 or batch!=32 or epochs!=1 or steps is not None or a.update_mode!="sequential" or a.data_regime!="fixed_total" or a.link_failure_rate!=0 or a.stale_view_rounds!=0): raise ValueError("canonical protocol requires D_max=4, canonical_er, sequential/full/fixed-total, no failures/staleness, intervals=5, batch=32, and one local epoch")
    if a.method=="morph" and MorphNode is None:
        raise ValueError(f"Morph requires morph.py exposing MorphNode: {MORPH_IMPORT_ERROR}")
    if a.method=="dissdl" and a.num_clients>a.dissdl_max_n: raise ValueError("DissDL disabled above --dissdl-max-n due to its all-client known-peer directory")
    if a.method=="fedavg" and initial=="bounded_connected": initial="canonical_er"
    return Config(a.method,a.num_clients,a.seed,rounds,a.protocol,a.alpha,a.dmax,dmax,topo,ev,initial,a.participation_rate,epochs,steps,batch,a.lr,a.output_dir,a.checkpoint_interval,a.checkpoint_path,a.resume,a.force,a.eval_clients,a.final_eval_all,a.data_root,a.update_mode,minimum,a.target_accuracy,a.graph_metric_interval,a.dissdl_max_n,a.num_workers,a.samples_per_client,a.link_failure_rate,a.stale_view_rounds,a.repair_warning_fraction,a.representation_mode,a.lfhe_start_round,data_regime=a.data_regime)

def main():
    try:
        if "--run-all" in sys.argv[1:]: return run_all(batch_parser().parse_args())
        return run(make_config(parser().parse_args()))
    except (ValueError,RuntimeError) as exc: print(f"error: {exc}",file=sys.stderr); return 2
if __name__=="__main__": raise SystemExit(main())
