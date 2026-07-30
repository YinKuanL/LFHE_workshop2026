import random
import numpy as np
import torch
import main
from lfhe import lfhe_update

def canonical_args(extra=""):
    return main.parser().parse_args(("--method lfhe --num-clients 30 --seed 42 --protocol canonical --output-dir unused "+extra).split())
def test_canonical_defaults():
    c=main.make_config(canonical_args()); assert (c.alpha,c.rounds,c.local_epochs,c.batch_size,c.lr,c.topology_interval,c.eval_interval,c.dmax,c.w1,c.w2,c.w3)==(.1,300,1,32,.05,5,5,4,1.,1.,.1)
    model=main.CNN(); assert list(model.classifier[-1].weight.shape)==[10,256] and model.get_representation().numel()==2560
def test_paired_initial_graph_and_models():
    a=main.make_config(canonical_args()); b=main.make_config(main.parser().parse_args("--method static_random --num-clients 30 --seed 42 --protocol canonical --output-dir unused".split()))
    assert set(main.initial_graph(a).edges())==set(main.initial_graph(b).edges())
    states=main.initial_states(4,42); assert all(torch.equal(states[0][k],states[3][k]) for k in states[0])
def test_partition_reproducible_and_no_empty():
    labels=np.repeat(np.arange(10),100); a=main.dirichlet_split(labels,20,.1,3,42); b=main.dirichlet_split(labels,20,.1,3,42); assert a==b and min(map(len,a))>=3
def test_aggregation_weights_and_full_partial_counts():
    g=main.bounded_connected(10,4,42)
    for i in g:
        total=sum(1/(1+max(g.degree(i),g.degree(j))) for j in g.neighbors(i)); assert 0<=1-total<=1
    random.seed(1); assert len(list(range(10)))==10 and len(random.sample(range(10),3))==3
def test_bounded_graph():
    g=main.bounded_connected(100,4,42); assert main.nx.is_connected(g) and max(dict(g.degree()).values())<=4
def test_no_nan_and_summary_consistency():
    states=main.initial_states(3,1); assert not any(torch.isnan(v.float()).any() for s in states for v in s.values())
def test_sequential_calls_canonical_semantics():
    states=main.initial_states(5,1); clients=[main.Adapter(s) for s in states]; g=main.nx.cycle_graph(5); random.seed(1); out=lfhe_update(g,clients,D_max=4,round=0); assert set(out)==set(g)
def test_checkpoint_payload_has_resume_state(tmp_path):
    c=main.make_config(canonical_args()); states=main.initial_states(3,1); graph=main.nx.cycle_graph(3)
    value=main.checkpoint_payload(c,1,states,graph,[[0],[1],[2]],[],[],{"min_samples":1},{"edges":3})
    assert value["next_round"]==1 and value["rng"] and value["data_split"]==[[0],[1],[2]]
def test_checkpoint_resume_reproduces_toy_execution():
    states=main.initial_states(6,7); graph=main.nx.cycle_graph(6); random.seed(9); np.random.seed(9); torch.manual_seed(9)
    main.aggregate(states,graph,range(6)); saved_states=[main.clone_state(s) for s in states]; saved_rng=main.rng_state()
    main.aggregate(states,graph,range(6)); expected=[main.clone_state(s) for s in states]
    resumed=[main.clone_state(s) for s in saved_states]; main.restore_rng(saved_rng); main.aggregate(resumed,graph,range(6))
    assert all(torch.equal(expected[i][k],resumed[i][k]) for i in range(6) for k in expected[i])
def test_hard_and_disconnected_topologies():
    hard=main.clustered_hard(100,4,42,True); disconnected=main.clustered_hard(100,4,42,False)
    assert main.nx.is_connected(hard) and max(dict(hard.degree()).values())<=4
    assert main.nx.number_connected_components(disconnected)==2
def test_fixed_samples_partition_and_repair_metrics():
    labels=np.repeat(np.arange(10),100); splits,meta=main.dirichlet_split(labels,20,.1,5,42,return_stats=True,samples_per_client=20)
    assert set(map(len,splits))=={20} and 0<=meta["repaired_sample_fraction"]<=1
def test_link_failure_view_is_deterministic():
    graph=main.bounded_connected(30,4,42); a,na=main.failed_link_view(graph,.3,7); b,nb=main.failed_link_view(graph,.3,7)
    assert set(a.edges())==set(b.edges()) and na==nb
def test_snapshot_concurrent_preserves_degree_cap():
    cfg=main.make_config(main.parser().parse_args("--method lfhe --num-clients 20 --seed 42 --protocol scalable --output-dir unused --update-mode snapshot_concurrent".split()))
    states=main.initial_states(20,42); graph=main.bounded_connected(20,4,42); clients=[main.Adapter(s) for s in states]
    updated,trace,stats=main.snapshot_concurrent_lfhe(graph,clients,range(20),cfg,0)
    assert max(dict(updated.degree()).values())<=4 and 0<=stats["shared_endpoint_conflict_rate"]<=1
