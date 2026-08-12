import csv
from pathlib import Path

import networkx as nx
import numpy as np

from lfhe_pac import (LFHEPACState, build_random_heterogeneous_state,
    discover_frozen_fof, enumerate_feasible_operations, feasible_operation_hash,
    graph_jaccard_swap_score, representation_swap_score, run_pac_epoch,
    select_one_proposal_per_initiator)

ROOT=Path(__file__).resolve().parents[1]

def state(n=50,seed=42):
    return build_random_heterogeneous_state(num_nodes=n,average_degree=3,dmax=4,seed=seed,edge_budget=round(n*3/2))

def snapshot(value, timestamp=0):
    reps={i:np.asarray([i, i*i%17],dtype=np.float64) for i in range(value.num_nodes)}
    return value.snapshot(reps,timestamp=timestamp)

def test_n50_initial_graph_and_protected_tree():
    value=state(); graph=value.graph; protected=nx.Graph(); protected.add_nodes_from(graph); protected.add_edges_from(value.protected_edges)
    assert graph.number_of_edges()==75 and nx.is_connected(graph) and max(dict(graph.degree()).values())<=4
    assert nx.is_tree(protected) and len(set(dict(graph.degree()).values()))>1 and value.edge_budget==75

def test_frozen_candidate_and_feasible_order_invariance():
    value=state(); snap=snapshot(value); forward=list(range(50)); reverse=forward[::-1]
    a=discover_frozen_fof(snap,candidate_budget=5,seed=42,initiator_order=forward)
    b=discover_frozen_fof(snap,candidate_budget=5,seed=42,initiator_order=reverse)
    assert a.stream_hash==b.stream_hash
    assert feasible_operation_hash(enumerate_feasible_operations(snap,a,initiator_order=forward))==feasible_operation_hash(enumerate_feasible_operations(snap,b,initiator_order=reverse))

def test_epoch_fixed_budget_lock_cleanup_and_checkpoint_roundtrip():
    value=state(); snap=snapshot(value); stream=discover_frozen_fof(snap,candidate_budget=5,seed=42)
    feasible=enumerate_feasible_operations(snap,stream); selected=select_one_proposal_per_initiator(feasible,method='lfhe_pac',seed=42)
    before=value.edge_count; result=run_pac_epoch(value,snap,selected,method='lfhe_pac',max_commits=12,seed=42)
    assert value.edge_count==before and not value.locks and nx.is_connected(value.graph) and max(dict(value.graph.degree()).values())<=4
    restored=LFHEPACState.restore(value.checkpoint()); assert restored.fingerprint()==value.fingerprint()
    assert result.committed_additions==0

def test_new_lfhe_and_random_fof_are_fixed_edge_swap_pair():
    for method,score_function in (
        ("random_fof_swap",None),
        ("lfhe_representation_swap",representation_swap_score),
    ):
        value=state(); snap=snapshot(value); before=value.edge_count
        stream=discover_frozen_fof(snap,candidate_budget=5,seed=42)
        feasible=enumerate_feasible_operations(snap,stream,**({"score_function":score_function} if score_function else {}))
        feasible=tuple(proposal for proposal in feasible if proposal.operation=="swap")
        selected=select_one_proposal_per_initiator(feasible,method=method,seed=42)
        result=run_pac_epoch(value,snap,selected,method=method,max_commits=12,seed=42,**({"score_function":score_function} if score_function else {}))
        assert all(proposal.operation=="swap" for proposal in feasible)
        assert value.edge_count==before and result.committed_additions==0
        assert nx.is_connected(value.graph) and max(dict(value.graph.degree()).values())<=4

def test_lfhe_expand_restores_historical_add_or_swap_path():
    value=build_random_heterogeneous_state(
        num_nodes=50, average_degree=3, dmax=4, seed=42, edge_budget=100
    )
    snap=snapshot(value, timestamp=5)
    stream=discover_frozen_fof(snap,candidate_budget=5,seed=42)
    feasible=enumerate_feasible_operations(snap,stream)
    additions=tuple(
        proposal for proposal in feasible
        if proposal.operation=="addition" and proposal.initiator_gain>0
        and proposal.has_negative_noninitiator_gain
    )
    assert additions
    selected=select_one_proposal_per_initiator(
        additions, method="lfhe_expand", seed=42
    )
    before=value.edge_count
    result=run_pac_epoch(
        value, snap, selected[:1], method="lfhe_expand", max_commits=2, seed=42
    )
    assert result.committed_additions==1
    assert result.committed_swaps==0
    assert value.edge_count==before+1
    assert nx.is_connected(value.graph)
    assert max(dict(value.graph.degree()).values())<=4

def test_lfhe_expand_is_separate_from_fixed_edge_lfhe_runner_mapping():
    from pac_methods import FIXED_EDGE_SWAP_METHODS, PAC_METHODS, PAC_PROTOCOL_METHOD
    assert PAC_PROTOCOL_METHOD["lfhe_expand"]=="lfhe_pac_v2"
    assert "lfhe_expand" in PAC_METHODS
    assert "lfhe_expand" not in FIXED_EDGE_SWAP_METHODS
    assert PAC_PROTOCOL_METHOD["lfhe"]=="lfhe_representation_swap"
    assert PAC_PROTOCOL_METHOD["random_fof"]=="random_fof_swap"
    assert PAC_PROTOCOL_METHOD["lfhe_pac"]=="lfhe_pac_strict"
    assert FIXED_EDGE_SWAP_METHODS==frozenset({"lfhe", "random_fof"})

def test_lfhe_expand_reuses_sparse_initial_family_with_historical_headroom():
    fixed=build_random_heterogeneous_state(
        num_nodes=100, average_degree=3, dmax=4, seed=42, edge_budget=150
    )
    expand=build_random_heterogeneous_state(
        num_nodes=100, average_degree=3, dmax=4, seed=42
    )
    assert set(fixed.graph.edges())==set(expand.graph.edges())
    assert fixed.edge_count==expand.edge_count==150
    assert fixed.edge_budget==150
    assert expand.edge_budget==200

def test_graph_jaccard_control_is_model_information_free():
    value=state(); graph=value.graph
    zeros={i:np.zeros(3,dtype=np.float64) for i in graph}
    random_reps={i:np.asarray([i+1,i*i+2,i%7+3],dtype=np.float64) for i in graph}
    assert graph_jaccard_swap_score(0,graph,zeros)==graph_jaccard_swap_score(0,graph,random_reps)

def test_manifest_contract():
    rows=list(csv.DictReader((ROOT/'manifests/workshop_lfhe_pac_main.csv').open()))
    assert len(rows)==20 and len({r['output_dir'] for r in rows})==20
    for n in (50,100,200,500):
        group=[r for r in rows if int(r['num_clients'])==n]
        assert len(group)==5 and {int(r['seed']) for r in group}==set(range(42,47))
    assert all(r['method']=='lfhe_pac' and r['rounds']=='300' and r['representation_mode']=='flatten' for r in rows)

def test_lfhe_expand_manifest_and_smoke_contracts():
    rows=list(csv.DictReader((ROOT/'manifests/workshop_lfhe_expand_n100.csv').open()))
    assert len(rows)==5 and {int(r['seed']) for r in rows}==set(range(42,47))
    assert len({r['output_dir'] for r in rows})==5
    expected={
        'method':'lfhe_expand','num_clients':'100','rounds':'300','protocol':'scalable',
        'alpha':'0.3','dmax':'4','degree_regime':'fixed4','topology_interval':'5',
        'eval_interval':'5','initial_graph':'bounded_connected','participation_rate':'1.0',
        'local_epochs':'1','batch_size':'32','lr':'0.05','checkpoint_policy':'auto',
        'final_eval_all':'true','update_mode':'sequential','data_regime':'fixed_total',
        'representation_mode':'flatten',
    }
    assert all(all(row[key]==value for key,value in expected.items()) for row in rows)
    smoke=list(csv.DictReader((ROOT/'manifests/workshop_lfhe_expand_n100_smoke.csv').open()))
    assert len(smoke)==1 and smoke[0]['seed']=='42' and smoke[0]['rounds']=='10'
    assert smoke[0]['method']=='lfhe_expand' and smoke[0]['num_clients']=='100'

def test_runner_schema_is_wired():
    source=(ROOT/'main.py').read_text()
    for field in ('pac_candidate_packets.jsonl','pac_proposals.jsonl','pac_epoch_summary.jsonl','candidate_stream_hash','all_pac_invariants_passed','protected_tree_unchanged'):
        assert field in source
