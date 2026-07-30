#!/usr/bin/env python3
"""Read-only approval gate for completed canonical or feasibility manifests."""
import argparse, csv, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--kind",choices=("canonical","feasibility"),required=True); p.add_argument("--historical-ranges",default="historical_canonical_ranges.json"); a=p.parse_args()
    rows=list(csv.DictReader(Path(a.manifest).open(encoding="utf-8"))); failures=[]
    historical=json.loads(Path(a.historical_ranges).read_text(encoding="utf-8")) if a.kind=="canonical" else None
    for row in rows:
        out=Path(row["output_dir"]); summary_path=out/"summary.json"; config_path=out/"config.json"
        if not summary_path.exists() or not config_path.exists() or not (out/"SUCCESS").exists(): failures.append(f"incomplete: {out}"); continue
        summary=json.loads(summary_path.read_text(encoding="utf-8")); config=json.loads(config_path.read_text(encoding="utf-8"))
        if a.kind=="canonical":
            expected=historical["methods"][row["method"]]; accuracy=summary.get("final_accuracy")
            if accuracy is None or not expected["min"]<=accuracy<=expected["max"]: failures.append(f"historical range failure: {out} accuracy={accuracy} expected={expected}")
            if config.get("representation_dimension")!=historical["representation_dimension"] or config.get("representation_shape")!=historical["representation_shape"]: failures.append(f"representation mismatch: {out}")
            required={"protocol":"canonical","participation_rate":1.0,"dmax":4,"update_mode":"sequential","data_regime":"fixed_total"}
            for key,value in required.items():
                if config.get(key)!=value: failures.append(f"canonical config mismatch {key}: {out}")
        elif not summary.get("feasibility_gate",{}).get("passed",False):
            failures.append(f"feasibility threshold failure: {out} checks={summary.get('feasibility_gate',{}).get('checks')}")
    report={"manifest":a.manifest,"kind":a.kind,"runs":len(rows),"passed":not failures,"failures":failures}
    print(json.dumps(report,indent=2)); return 0 if not failures else 2
if __name__=="__main__": raise SystemExit(main())
