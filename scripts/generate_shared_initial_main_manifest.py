#!/usr/bin/env python3
"""Generate the clean main suite with one Static-Random initial topology."""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests" / "workshop_main_fresh_md_aligned_n50_500.csv"
TARGET = ROOT / "manifests" / "workshop_main_shared_static_init_n50_500.csv"
SUITE = "workshop_main_shared_static_init_n50_500"


def generate() -> list[dict[str, str]]:
    with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for source in rows:
        row = dict(source)
        row["suite"] = SUITE
        row["shared_initial_topology"] = "true"
        run_name = Path(row["output_dir"]).name
        row["output_dir"] = f"outputs/{SUITE}/{run_name}_sharedstaticinitv1"
        result.append(row)
    if len(result) != 120:
        raise ValueError(f"expected 120 runs, got {len(result)}")
    if {row["method"] for row in result} != {
        "static_random", "epidemic", "dissdl", "random_fof", "morph", "lfhe"
    }:
        raise ValueError("unexpected method set")
    if len({row["output_dir"] for row in result}) != len(result):
        raise ValueError("duplicate output directory")
    return result


def render(rows: list[dict[str, str]]) -> str:
    fields = list(rows[0])
    fields.remove("shared_initial_topology")
    fields.insert(fields.index("output_dir"), "shared_initial_topology")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render(generate())
    if args.check:
        if not TARGET.exists() or TARGET.read_text(encoding="utf-8-sig") != expected:
            raise SystemExit(f"out of date: {TARGET}")
    else:
        TARGET.write_text(expected, encoding="utf-8", newline="")
        print(f"wrote {TARGET} with 120 runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
