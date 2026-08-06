import csv
from pathlib import Path

import networkx as nx
import numpy as np

from lfhe_pac import (LFHEPACState, build_random_heterogeneous_state,
    discover_frozen_fof, enumerate_feasible_operations, feasible_operation_hash,
    run_pac_epoch, select_one_proposal_per_initiator)

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

def test_manifest_contract():
    rows=list(csv.DictReader((ROOT/'manifests/workshop_lfhe_pac_main.csv').open()))
    assert len(rows)==20 and len({r['output_dir'] for r in rows})==20
    for n in (50,100,200,500):
        group=[r for r in rows if int(r['num_clients'])==n]
        assert len(group)==5 and {int(r['seed']) for r in group}==set(range(42,47))
    assert all(r['method']=='lfhe_pac' and r['rounds']=='300' and r['representation_mode']=='flatten' for r in rows)

def test_runner_schema_is_wired():
    source=(ROOT/'main.py').read_text()
    for field in ('pac_candidate_packets.jsonl','pac_proposals.jsonl','pac_epoch_summary.jsonl','candidate_stream_hash','all_pac_invariants_passed','protected_tree_unchanged'):
        assert field in source
