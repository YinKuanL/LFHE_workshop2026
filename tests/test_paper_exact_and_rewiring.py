import csv,random
from pathlib import Path
import networkx as nx
import numpy as np
import torch
import main
import main_paper_exact as paper

ROOT=Path(__file__).resolve().parents[1]

def test_paper_model_and_representation():
    model=paper.CNN(); assert model.classifier[-1].weight.shape==(10,256); assert model.get_representation().shape==(10,)
def test_paper_dmax():
    for n in (10,50,100): assert paper.compute_dmax(n)==max(2,int(2*np.log(n)))
def test_paper_dirichlet_exact_no_repair():
    labels=np.repeat(np.arange(10),100); a=paper.dirichlet_split(labels,20,.3,42); b=paper.dirichlet_split(labels,20,.3,42)
    assert a==b and sorted(x for split in a for x in split)==list(range(1000))
def test_paper_distinct_sequential_initial_weights():
    torch.manual_seed(42); a=paper.CNN(); b=paper.CNN(); assert not torch.equal(a.classifier[-1].weight,b.classifier[-1].weight)
    workshop=main.initial_states(2,42); assert torch.equal(workshop[0]['classifier.4.weight'],workshop[1]['classifier.4.weight'])
def test_paper_er_is_deterministic_and_matched():
    assert paper.connected_er(30,42).edges()==paper.connected_er(30,42).edges()
def test_paper_epidemic_and_dissdl_degrees():
    assert set(dict(paper.build_epidemic_graph(20,4,42).out_degree()).values())=={4}
    random.seed(42); assert set(map(len,paper.dissdl_initial(20,3).values()))=={3}
def test_bounded_connected_v2_headroom_and_determinism():
    a=main.bounded_connected(100,4,42); b=main.bounded_connected(100,4,42)
    rng=random.Random(42); order=list(range(100)); rng.shuffle(order); old=nx.Graph(); old.add_nodes_from(range(100))
    for i,u in enumerate(order): old.add_edge(u,order[(i+1)%100])
    pairs=[(u,v) for u in range(100) for v in range(u+1,100)]; rng.shuffle(pairs)
    for u,v in pairs:
        if old.degree(u)<4 and old.degree(v)<4 and not old.has_edge(u,v): old.add_edge(u,v)
    old_slots=sum(4-d for _,d in old.degree())
    assert set(a.edges())==set(b.edges()) and nx.is_connected(a) and max(dict(a.degree()).values())<=4
    info=main.adaptive_topology_preflight(a,4); assert info['initial_available_degree_slots']>old_slots and info['initial_fraction_nodes_below_cap']>=.25 and 2.9<=sum(dict(a.degree()).values())/100<=3.1
def test_random_fof_addition_and_counters(monkeypatch):
    g=nx.cycle_graph(8); monkeypatch.setattr(random,'choice',lambda values:list(values)[0]); monkeypatch.setattr(random,'random',lambda:0.)
    out,trace=main.fof_update(g,[0],4,True); assert out.number_of_edges()==g.number_of_edges()+1; assert sum(e['action'].endswith('accepted_addition') for e in trace)==1; assert max(dict(out.degree()).values())<=4
def test_random_fof_swap_connected(monkeypatch):
    g=nx.cycle_graph(8); g.add_edges_from([(0,2),(0,6)]); monkeypatch.setattr(random,'choice',lambda values:2 if 2 in values else list(values)[0]); monkeypatch.setattr(random,'random',lambda:0.)
    out,trace=main.fof_update(g,[0],4,True); assert any(e['action']=='accepted_swap' for e in trace); assert nx.is_connected(out) and max(dict(out.degree()).values())<=4
def test_saturated_candidate_rejected(monkeypatch):
    g=nx.complete_graph(5); monkeypatch.setattr(random,'choice',lambda values:list(values)[0]); out,trace=main.fof_update(g,[0],4,True); assert set(out.edges())==set(g.edges()) and not any(e['action'].startswith('accepted') for e in trace)
def test_large_morph_checkpoint_policy():
    cfg=main.make_config(main.parser().parse_args('--method morph --num-clients 500 --seed 42 --protocol scalable --output-dir unused'.split()))
    assert main.checkpoint_policy(cfg)[0]=='disabled' and main.checkpoint_policy(cfg)[2] is False
    assert not main.should_checkpoint(cfg,10,True,True)
def test_protocol_output_separation_and_manifest_count():
    rows=list(csv.DictReader((ROOT/'manifests/paper_exact.csv').open())); assert len(rows)==54
    assert all(r['output_dir'].startswith('outputs/paper_exact/') for r in rows)
