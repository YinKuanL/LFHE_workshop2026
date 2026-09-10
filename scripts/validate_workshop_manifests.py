#!/usr/bin/env python3
"""Validate generated workshop manifests without importing training dependencies."""
from __future__ import annotations
import csv,importlib.util,io,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("generator",ROOT/"scripts/generate_workshop_manifests.py")
generator=importlib.util.module_from_spec(spec); spec.loader.exec_module(generator)

def main():
    expected=generator.generate(); generator.validate(expected); all_outputs=set()
    for name,count in generator.EXPECTED.items():
        path=ROOT/"manifests"/f"{name}.csv"; rows=list(csv.DictReader(path.open(encoding="utf-8-sig")))
        assert len(rows)==count, (name,len(rows),count)
        assert set(rows[0])==set(generator.FIELDS)
        for row in rows:
            assert row["output_dir"] not in all_outputs and "paper_exact" not in row["output_dir"]
            all_outputs.add(row["output_dir"])
            assert row["method"] in ("static_random","epidemic","dissdl","random_fof","morph","lfhe")
            assert row["protocol"]=="scalable" and row["update_mode"] in ("sequential","snapshot_concurrent")
            assert row["initial_graph"] in ("bounded_connected","ring","clustered","disconnected_clusters")
            if row["initial_graph"]=="bounded_connected": assert "initheadroomv2" in row["output_dir"]
            if row["method"]=="morph" and int(row["num_clients"])>=500: assert row["checkpoint_policy"]=="disabled"
            if row["data_regime"]=="fixed_samples_per_client":
                total=int(row["num_clients"])*int(row["samples_per_client"]); assert total<=50000
        assert rows==list(csv.DictReader(io.StringIO(generator.text(expected[name])))), f"generated content mismatch: {name}"
    headline=list(csv.DictReader((ROOT/'manifests/workshop_headline.csv').open()))
    assert len(headline)==120 and {int(r['dmax']) for r in headline}=={4} and {int(r['seed']) for r in headline}=={42,43,44,45,46}
    full={r['output_dir'] for r in csv.DictReader((ROOT/'manifests/workshop_large_scale.csv').open())}
    feasibility={r['output_dir'] for r in csv.DictReader((ROOT/'manifests/workshop_large_scale_feasibility.csv').open())}; assert full.isdisjoint(feasibility)
    degree=list(csv.DictReader((ROOT/'manifests/workshop_degree_sweep.csv').open()))
    keys=[(r['method'],r['num_clients'],r['seed'],r['dmax']) for r in degree]; assert len(keys)==len(set(keys))
    fixed=list(csv.DictReader((ROOT/'manifests/workshop_fixed_per_client.csv').open()))
    assert {int(r['num_clients'])*int(r['samples_per_client']) for r in fixed}=={10000,20000,50000}
    pac=list(csv.DictReader((ROOT/'manifests/workshop_lfhe_pac_main.csv').open()))
    assert len(pac)==20
    assert {int(r['num_clients']) for r in pac}=={50,100,200,500}
    for n in (50,100,200,500):
        group=[r for r in pac if int(r['num_clients'])==n]; assert len(group)==5 and {int(r['seed']) for r in group}==set(range(42,47))
    assert len({r['output_dir'] for r in pac})==20 and all(r['method']=='lfhe_pac' and r['rounds']=='300' and r['alpha']=='0.3' and r['dmax']=='4' and r['data_regime']=='fixed_total' and r['participation_rate']=='1.0' and r['final_eval_all']=='true' for r in pac)
    print("validated LFHE-PAC: 20 rows; N=50/100/200/500 each 5; seeds=42-46; unique outputs; method=lfhe_pac; rounds=300; alpha=0.3; dmax=4; fixed_total; full participation; final eval all")
    print("validated",sum(generator.EXPECTED.values()),"workshop rows across",len(generator.EXPECTED),"manifests")
if __name__=="__main__": main()
