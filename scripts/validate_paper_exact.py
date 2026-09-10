#!/usr/bin/env python3
import csv,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    errors=[]; rows=list(csv.DictReader((ROOT/'manifests/paper_exact.csv').open(encoding='utf-8')))
    for row in rows:
        out=ROOT/row['output_dir']; summary=out/'summary.json'
        if not (out/'SUCCESS').exists() or not summary.exists(): errors.append(f'missing: {row}'); continue
        s=json.loads(summary.read_text()); expected={'protocol':'paper_exact','alpha':.3,'rounds':300,'local_epochs':1,'batch_size':32,'lr':.05,'topology_interval':5,'evaluation_interval':5,'representation':'classifier[-1].weight.mean(dim=1)','client_initialization':'distinct_sequential_models','partition_repair':False,'all_clients_evaluated':True}
        for k,v in expected.items():
            if s.get(k)!=v: errors.append(f'{summary}: {k} mismatch')
        if s.get('D_max')!=max(2,int(2*math.log(int(row['num_clients'])))): errors.append(f'{summary}: bad D_max')
        if row['method']=='epidemic' and s.get('epidemic_s')!=4: errors.append(f'{summary}: epidemic s')
        if row['method']=='dissdl' and s.get('dissdl_initial_degree')!=3: errors.append(f'{summary}: DissDL degree')
    for n in ('10','50','100'):
        for seed in ('42','43','44'):
            a=ROOT/f'outputs/paper_exact/random_n{n}_seed{seed}/graph_initial.edgelist'; b=ROOT/f'outputs/paper_exact/lfhe_n{n}_seed{seed}/graph_initial.edgelist'
            if a.exists() and b.exists() and a.read_bytes()!=b.read_bytes(): errors.append(f'initial graph mismatch N={n} seed={seed}')
    print(*errors,sep='\n'); return bool(errors)
if __name__=='__main__': raise SystemExit(main())
