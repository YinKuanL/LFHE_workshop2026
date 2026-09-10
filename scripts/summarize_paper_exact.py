#!/usr/bin/env python3
import csv,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    rows=list(csv.DictReader((ROOT/'manifests/paper_exact.csv').open(encoding='utf-8'))); runs=[]
    for row in rows:
        summary=ROOT/row['output_dir']/'summary.json'; value=json.loads(summary.read_text()) if summary.exists() else {}
        runs.append({**row,'status':value.get('status','missing'),'final_accuracy':value.get('final_accuracy','')})
    with (ROOT/'paper_exact_runs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=runs[0]); w.writeheader(); w.writerows(runs)
    groups={}
    for r in runs:
        if r['final_accuracy']!='': groups.setdefault((r['method'],r['num_clients']),[]).append(float(r['final_accuracy']))
    output=[]
    for (method,n),values in sorted(groups.items()): output.append({'method':method,'num_clients':n,'completed_seeds':len(values),'mean':statistics.mean(values),'sample_std':statistics.stdev(values) if len(values)>1 else math.nan})
    with (ROOT/'paper_exact_aggregated.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['method','num_clients','completed_seeds','mean','sample_std']); w.writeheader(); w.writerows(output)
    missing=sum(r['status']=='missing' for r in runs); print(f'{len(runs)-missing}/54 complete; {missing} missing')
if __name__=='__main__': main()
