#!/usr/bin/env python3
"""Isolated reproduction of the historical main-paper scalability runner."""
from __future__ import annotations
import argparse, copy, json, math, os, random, time
from pathlib import Path
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from epidemic import build_epidemic_graph
from lfhe import lfhe_update
from morph import MorphNode

METHODS=("ring","random","epidemic","dissdl","morph","lfhe")
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(True),
            nn.Conv2d(32,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(True),
            nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(True),nn.MaxPool2d(2))
        self.avgpool=nn.AdaptiveAvgPool2d((4,4))
        self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(1024,256),nn.ReLU(True),nn.Dropout(.3),nn.Linear(256,10))
    def forward(self,x): return self.classifier(self.avgpool(self.features(x)))
    def get_representation(self): return self.classifier[-1].weight.data.mean(dim=1).flatten()

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def load_dataset(root):
    from torchvision import datasets,transforms
    norm=transforms.Normalize([.4914,.4822,.4465],[.2023,.1994,.2010])
    train_tf=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),norm])
    test_tf=transforms.Compose([transforms.ToTensor(),norm])
    return datasets.CIFAR10(root,train=True,download=False,transform=train_tf),datasets.CIFAR10(root,train=False,download=False,transform=test_tf)

def dirichlet_split(labels,num_clients,alpha,seed):
    np.random.seed(seed); result=[[] for _ in range(num_clients)]
    for c in range(int(np.max(labels))+1):
        idx=np.where(np.asarray(labels)==c)[0]; np.random.shuffle(idx)
        cuts=(np.cumsum(np.random.dirichlet(np.repeat(alpha,num_clients)))*len(idx)).astype(int)[:-1]
        for i,part in enumerate(np.split(idx,cuts)): result[i].extend(part.tolist())
    return result

class Client:
    def __init__(self,dataset,indices,lr=.05,batch_size=32):
        if not indices: raise RuntimeError("paper_exact produced an empty client")
        self.loader=DataLoader(Subset(dataset,indices),batch_size=batch_size,shuffle=True)
        self.model=CNN().to(DEVICE); self.optimizer=optim.SGD(self.model.parameters(),lr=lr); self.criterion=nn.CrossEntropyLoss()
    def train(self,epochs=1):
        self.model.train()
        for _ in range(epochs):
            for x,y in self.loader:
                x,y=x.to(DEVICE),y.to(DEVICE); self.optimizer.zero_grad(); loss=self.criterion(self.model(x),y); loss.backward(); self.optimizer.step()
    def evaluate(self,loader):
        self.model.eval(); correct=total=0; loss=0.
        with torch.no_grad():
            for x,y in loader:
                x,y=x.to(DEVICE),y.to(DEVICE); out=self.model(x); loss+=self.criterion(out,y).item(); correct+=int((out.argmax(1)==y).sum()); total+=len(y)
        return correct/total,loss/len(loader)

def compute_dmax(n): return max(2,int(2*np.log(n)))
def connected_er(n,seed):
    rng=np.random.RandomState(seed); p=4/(n-1)
    while True:
        graph=nx.erdos_renyi_graph(n,p,seed=int(rng.randint(0,1_000_000)))
        if nx.is_connected(graph): return graph
def ring(n): return nx.cycle_graph(n)
def dissdl_initial(n,degree=3):
    return {i:random.sample([j for j in range(n) if j!=i],degree) for i in range(n)}

def average_models(clients,graph):
    updates=[]
    for i,client in enumerate(clients):
        values=[]; total=0.
        for j in graph.neighbors(i):
            w=1/(1+max(graph.degree(i),graph.degree(j))); total+=w; values.append((w,clients[j].model.state_dict()))
        values.append((1-total,client.model.state_dict())); updates.append({k:sum(w*s[k] for w,s in values) for k in values[0][1]})
    for client,state in zip(clients,updates): client.model.load_state_dict(state)

def directed_average(clients,graph):
    updates=[]
    for i,client in enumerate(clients):
        incoming=list(graph.predecessors(i)); states=[clients[j].model.state_dict() for j in incoming]+[client.model.state_dict()]
        updates.append({k:sum(s[k] for s in states)/len(states) for k in states[0]})
    for client,state in zip(clients,updates): client.model.load_state_dict(state)

class HistoricalDissDLNode:
    def __init__(self,node_id,model,neighbors):
        self.id=node_id; self.model=model; self.wanted_senders=set(neighbors); self.known_peers=set(); self.received_models={}
    def aggregate(self):
        models=[self.model]+[self.received_models[i] for i in self.wanted_senders if i in self.received_models]
        states=[m.state_dict() for m in models]; self.model.load_state_dict({k:sum(s[k] for s in states)/len(states) for k in states[0]})
    def update_wanted_senders(self):
        candidates=sorted(self.known_peers-self.wanted_senders-{self.id})
        if candidates and self.wanted_senders: self.wanted_senders.remove(random.choice(sorted(self.wanted_senders))); self.wanted_senders.add(random.choice(candidates))

def morph_nodes(clients,seed,degree=4):
    rng=random.Random(seed); nodes=[]
    for i,c in enumerate(clients):
        peers=[j for j in range(len(clients)) if j!=i]
        nodes.append(MorphNode(i,c.model,rng.sample(peers,degree),in_degree=degree,beta=500.,change_iter=5,seed=seed,indirect_history_k=5,device=DEVICE))
    return nodes
def morph_step(nodes,rnd):
    for n in nodes: n.begin_round(rnd); n.update_wanted_senders(rnd); n.validate_state()
    payloads={n.id:n.build_model_payload(degree=len(n.wanted_senders)) for n in nodes}
    for receiver in nodes:
        for sender in sorted(receiver.requested_senders()):
            if nodes[sender].should_send_to(receiver.id,True): receiver.receive_model_payload(sender,payloads[sender])
    for n in nodes: n.aggregate()
def morph_graph(nodes):
    g=nx.DiGraph(); g.add_nodes_from(range(len(nodes)))
    for n in nodes: g.add_edges_from((s,n.id) for s in n.wanted_senders)
    return g

def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(value,indent=2),encoding="utf-8"); os.replace(tmp,path)
def write_edges(path,graph): nx.write_edgelist(graph,path,data=False)
def valid_success(out):
    try: return (out/"SUCCESS").exists() and json.loads((out/"summary.json").read_text())["protocol"]=="paper_exact"
    except (OSError,KeyError,json.JSONDecodeError): return False

def run(args):
    out=Path(args.output_dir)
    if valid_success(out) and not args.force: print(f"[skip] {out}"); return 0
    if out.exists() and any(out.iterdir()) and not args.force: raise RuntimeError("output directory is non-empty; use --force or a unique path")
    out.mkdir(parents=True,exist_ok=True); set_seed(args.seed); train,test=load_dataset(args.data_root); splits=dirichlet_split(train.targets,args.num_clients,.3,args.seed)
    empty=[i for i,s in enumerate(splits) if not s]
    if empty: raise RuntimeError(f"empty clients for seed={args.seed}, N={args.num_clients}: {empty}")
    clients=[Client(train,s) for s in splits]; loader=DataLoader(test,batch_size=256,shuffle=False)
    if args.method in ("random","lfhe"): graph=connected_er(args.num_clients,args.seed)
    elif args.method=="ring": graph=ring(args.num_clients)
    elif args.method=="epidemic": graph=build_epidemic_graph(args.num_clients,s=4,seed=args.seed)
    elif args.method=="dissdl": graph=dissdl_initial(args.num_clients,3)
    else: graph=None
    diss=[]
    if args.method=="dissdl":
        diss=[HistoricalDissDLNode(i,clients[i].model,graph[i]) for i in range(args.num_clients)]
        for n in diss: n.known_peers=set(range(args.num_clients))-{n.id}
        initial_graph=nx.DiGraph((s,i) for i,senders in graph.items() for s in senders)
    elif args.method=="morph": nodes=morph_nodes(clients,args.seed); initial_graph=morph_graph(nodes); graph=initial_graph
    else: initial_graph=graph.copy()
    write_edges(out/"graph_initial.edgelist",initial_graph)
    config={"protocol":"paper_exact","method":args.method,"num_clients":args.num_clients,"seed":args.seed,"rounds":args.rounds,"alpha":.3,"batch_size":32,"local_epochs":1,"lr":.05,"topology_interval":5,"evaluation_interval":5,"D_max":compute_dmax(args.num_clients),"representation":"classifier[-1].weight.mean(dim=1)","client_initialization":"distinct_sequential_models","partition_repair":False,"all_clients_evaluated":True,"implementation_source":"historical_main_scalability_semantics","epidemic_s":4,"dissdl_initial_degree":3}
    atomic_json(out/"config.json",config); records=[]; started=time.time()
    for rnd in range(args.rounds):
        for c in clients: c.train(1)
        if args.method=="epidemic": graph=build_epidemic_graph(args.num_clients,s=4,seed=args.seed+rnd); directed_average(clients,graph)
        elif args.method=="dissdl":
            for node in diss: node.received_models={i:copy.deepcopy(clients[i].model) for i in node.wanted_senders}
            for node in diss:
                node.aggregate(); clients[node.id].model.load_state_dict(node.model.state_dict())
                if rnd%5==0: node.update_wanted_senders()
            graph=nx.DiGraph((s,n.id) for n in diss for s in n.wanted_senders)
        elif args.method=="morph": morph_step(nodes,rnd); graph=morph_graph(nodes)
        else:
            average_models(clients,graph)
            if args.method=="lfhe" and rnd%5==0: graph=lfhe_update(graph,clients,.05,compute_dmax(args.num_clients),1.,1.,.1,rnd)
        rec={"round":rnd}
        if rnd%5==0 or rnd==args.rounds-1:
            values=[c.evaluate(loader) for c in clients]; params=torch.stack([torch.cat([p.data.flatten().cpu() for p in c.model.parameters()]) for c in clients]); mean=params.mean(0)
            rec.update(mean_accuracy=float(np.mean([x[0] for x in values])),mean_loss=float(np.mean([x[1] for x in values])),inter_node_variance=float(((params-mean).norm(dim=1)**2).mean()),evaluated_clients=len(clients))
        records.append(rec)
        with (out/"metrics.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(rec)+"\n")
    evals=[r for r in records if "mean_accuracy" in r]; write_edges(out/"graph_final.edgelist",graph)
    summary={**config,"N":args.num_clients,"official_paper_scale":args.num_clients in (10,50,100),"paper_extension":args.num_clients==500,"final_accuracy":evals[-1]["mean_accuracy"],"accuracy_evaluation_rounds":[r["round"] for r in evals],"accuracy_curve":[r["mean_accuracy"] for r in evals],"loss_curve":[r["mean_loss"] for r in evals],"inter_node_variance":[r["inter_node_variance"] for r in evals],"wall_clock_seconds":time.time()-started,"status":"complete"}
    atomic_json(out/"summary.json",summary); (out/"SUCCESS").write_text("SUCCESS\n",encoding="utf-8"); return 0

def parser():
    p=argparse.ArgumentParser(); p.add_argument("--method",choices=METHODS,required=True); p.add_argument("--num-clients",type=int,required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--output-dir",required=True); p.add_argument("--data-root",default=os.getenv("LFHE_DATA_ROOT","./data")); p.add_argument("--force",action="store_true"); p.add_argument("--rounds",type=int,default=300,help=argparse.SUPPRESS); return p
def main():
    try: return run(parser().parse_args())
    except (ValueError,RuntimeError) as exc: print(f"error: {exc}",file=__import__('sys').stderr); return 2
if __name__=="__main__": raise SystemExit(main())
