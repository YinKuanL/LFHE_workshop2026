import csv,importlib.util,math
from pathlib import Path
import networkx as nx
import pytest
import main

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('workshop_generator',ROOT/'scripts/generate_workshop_manifests.py')
generator=importlib.util.module_from_spec(spec); spec.loader.exec_module(generator)

def rows(name): return list(csv.DictReader((ROOT/f'manifests/{name}.csv').open()))

def test_expected_counts_and_unique_outputs():
    outputs=[]
    for name,count in generator.EXPECTED.items():
        values=rows(name); assert len(values)==count; outputs.extend(v['output_dir'] for v in values)
    assert len(outputs)==len(set(outputs)) and not any('paper_exact' in x for x in outputs)

def test_headline_and_large_scale_separation():
    headline=rows('workshop_headline'); assert len(headline)==120 and {int(x['dmax']) for x in headline}=={4}
    assert {int(x['seed']) for x in headline}=={42,43,44,45,46}
    assert {x['output_dir'] for x in rows('workshop_large_scale')}.isdisjoint(x['output_dir'] for x in rows('workshop_large_scale_feasibility'))

def test_fixed_client_capacity_and_degree_deduplication():
    fixed=rows('workshop_fixed_per_client'); assert {int(x['num_clients'])*int(x['samples_per_client']) for x in fixed}=={10000,20000,50000}
    degree=rows('workshop_degree_sweep'); keys=[(x['method'],x['num_clients'],x['seed'],x['dmax']) for x in degree]
    assert len(keys)==len(set(keys)) and len(degree)==99

def test_restrictive_preflight_and_ordinary_failfast():
    saturated=nx.cycle_graph(20); info=main.adaptive_topology_preflight(saturated,2,fail=False); assert not info['structurally_rewirable']
    with pytest.raises(RuntimeError): main.adaptive_topology_preflight(saturated,2,fail=True)

def test_hard_disconnected_and_paired_initial_graphs():
    graph=main.clustered_hard(200,4,42,False); assert nx.number_connected_components(graph)==2
    base='--num-clients 200 --seed 42 --protocol scalable --dmax 4 --initial-graph bounded_connected --output-dir unused'
    a=main.make_config(main.parser().parse_args(('--method random_fof '+base).split())); b=main.make_config(main.parser().parse_args(('--method lfhe '+base).split()))
    assert set(main.initial_graph(a).edges())==set(main.initial_graph(b).edges())

def test_concurrent_metric_schema_and_bounded_headroom():
    source=(ROOT/'main.py').read_text(); required=('proposals_generated','accepted_proposals','endpoint_conflicts','degree_conflicts','stale_proposal_rejections','connectivity_safeguard_rejections','temporary_degree_violations','final_degree_violations')
    assert all(name in source for name in required)
    a=main.bounded_connected(100,4,42); b=main.bounded_connected(100,4,42)
    assert set(a.edges())==set(b.edges()) and sum(4-d for _,d in a.degree())>0

def test_slurm_worker_contract_and_large_morph_policy():
    text=(ROOT/'slurm/run_workshop_manifest_mira.sbatch').read_text(); assert 'cd "$HOME/LFHE_workshop"' in text and 'dirname "$0"' not in text and '#SBATCH --array' not in text
    for name in ('workshop_headline','workshop_large_scale','workshop_large_scale_feasibility'):
        for value in rows(name):
            if value['method']=='morph' and int(value['num_clients'])>=500: assert value['checkpoint_policy']=='disabled'

def test_all_rows_parse_to_config():
    for name in generator.EXPECTED:
      for value in rows(name):
        args=['--method',value['method'],'--num-clients',value['num_clients'],'--seed',value['seed'],'--rounds',value['rounds'],'--protocol',value['protocol'],'--alpha',value['alpha'],'--dmax',value['dmax'],'--degree-regime',value['degree_regime'],'--topology-interval',value['topology_interval'],'--eval-interval',value['eval_interval'],'--initial-graph',value['initial_graph'],'--participation-rate',value['participation_rate'],'--local-epochs',value['local_epochs'],'--batch-size',value['batch_size'],'--lr',value['lr'],'--checkpoint-interval',value['checkpoint_interval'],'--checkpoint-policy',value['checkpoint_policy'],'--eval-clients',value['eval_clients'],'--update-mode',value['update_mode'],'--data-regime',value['data_regime'],'--link-failure-rate',value['link_failure_rate'],'--stale-view-rounds',value['stale_view_rounds'],'--representation-mode',value['representation_mode'],'--lfhe-start-round',value['lfhe_start_round'],'--output-dir',value['output_dir']]
        if value['samples_per_client']: args += ['--samples-per-client',value['samples_per_client']]
        if value['min_samples_per_client']: args += ['--min-samples-per-client',value['min_samples_per_client']]
        cfg=main.make_config(main.parser().parse_args(args)); assert cfg.output_dir==value['output_dir']
