import json
from pathlib import Path

from scripts.build_topology_animation import collect_runs, render_html


def test_topology_animation_reconstructs_selectable_run(tmp_path: Path):
    run = tmp_path / "lfhe_n4_seed42"
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps({"method": "lfhe", "num_clients": 4, "seed": 42, "dmax": 3}),
        encoding="utf-8",
    )
    (run / "graph_initial.edgelist").write_text("0 1\n1 2\n2 3\n", encoding="utf-8")
    (run / "pac_edge_deltas.jsonl").write_text(
        json.dumps({"round": 5, "added": [[0, 2]], "removed": [[1, 2]]}) + "\n",
        encoding="utf-8",
    )
    rows = collect_runs(tmp_path)
    assert len(rows) == 1
    assert rows[0]["m"] == "lfhe"
    assert rows[0]["n"] == 4
    assert rows[0]["s"] == 42
    assert rows[0]["frames"] == [{"r": 5, "a": [[0, 2]], "d": [[1, 2]]}]
    page = render_html(rows, "Topology evolution")
    assert "Method<select id=\"method\"" in page
    assert "N<select id=\"n\"" in page
    assert "Seed<select id=\"seed\"" in page
    assert "Client neighborhood" in page
    assert "Topology only" in page
    assert "Highlight changes" in page
    assert "added this update" in page


def test_topology_animation_ignores_incomplete_jsonl_tail(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps({"method": "random_fof", "num_clients": 3, "seed": 43, "dmax": 2}),
        encoding="utf-8",
    )
    (run / "graph_initial.edgelist").write_text("0 1\n1 2\n", encoding="utf-8")
    (run / "pac_edge_deltas.jsonl").write_text(
        '{"round":0,"added":[],"removed":[]}\n{"round":', encoding="utf-8"
    )
    rows = collect_runs(tmp_path)
    assert len(rows) == 1
    assert [frame["r"] for frame in rows[0]["frames"]] == [0]
