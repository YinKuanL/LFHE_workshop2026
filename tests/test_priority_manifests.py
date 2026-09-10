import csv
from collections import Counter
from pathlib import Path

from pac_methods import FIXED_EDGE_SWAP_METHODS, PAC_PROTOCOL_METHOD


ROOT = Path(__file__).resolve().parents[1]


def _rows(name):
    with (ROOT / "manifests" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _assert_common(rows):
    assert len({row["output_dir"] for row in rows}) == len(rows)
    assert all(row["rounds"] == "300" for row in rows)
    assert all(row["alpha"] == "0.3" for row in rows)
    assert all(row["dmax"] == "4" for row in rows)
    assert all(row["degree_regime"] == "fixed4" for row in rows)
    assert all(row["topology_interval"] == "5" for row in rows)
    assert all(row["eval_interval"] == "5" for row in rows)
    assert all(row["local_epochs"] == "1" for row in rows)
    assert all(row["batch_size"] == "32" for row in rows)
    assert all(row["lr"] == "0.05" for row in rows)
    assert all(row["representation_mode"] == "flatten" for row in rows)
    assert all(row["update_mode"] == "sequential" for row in rows)
    assert all(row["final_eval_all"] == "true" for row in rows)


def test_p1_fixed_samples_manifests():
    full = _rows("workshop_p1_fixed_samples_per_client.csv")
    short = _rows("workshop_p1_fixed_samples_n100_500.csv")
    _assert_common(full)
    _assert_common(short)
    assert len(full) == 27
    assert len(short) == 18
    assert Counter((int(r["num_clients"]), r["method"]) for r in full) == {
        (n, method): 3
        for n in (100, 500, 1000)
        for method in ("static_random", "random_fof", "lfhe")
    }
    assert {int(r["seed"]) for r in full} == {42, 43, 44}
    assert all(r["data_regime"] == "fixed_per_client" for r in full)
    assert all(r["samples_per_client"] == "25" for r in full)
    assert all(r["min_samples_per_client"] == "25" for r in full)


def test_p4_disconnected_fof_manifest():
    rows = _rows("workshop_p4_disconnected_fof.csv")
    _assert_common(rows)
    assert len(rows) == 6
    assert {r["method"] for r in rows} == {"random_fof", "lfhe"}
    assert {int(r["seed"]) for r in rows} == {42, 43, 44}
    assert all(r["num_clients"] == "100" for r in rows)
    assert all(r["initial_graph"] == "disconnected_clusters" for r in rows)
    assert all(r["participation_rate"] == "1.0" for r in rows)


def test_p4_dropout_manifest_uses_partial_participation():
    rows = _rows("workshop_p4_dropout_n200.csv")
    _assert_common(rows)
    assert len(rows) == 36
    assert Counter((r["participation_rate"], r["method"]) for r in rows) == {
        (rate, method): 3
        for rate in ("1.0", "0.9", "0.7", "0.5")
        for method in ("static_random", "random_fof", "lfhe")
    }
    assert {int(r["seed"]) for r in rows} == {42, 43, 44}
    assert all(r["num_clients"] == "200" for r in rows)
    assert all(r["initial_graph"] == "bounded_connected" for r in rows)
    assert all(r["data_regime"] == "fixed_total" for r in rows)


def test_priority_manifests_use_md_aligned_fixed_edge_methods():
    assert PAC_PROTOCOL_METHOD["lfhe"] == "lfhe_representation_swap"
    assert {"lfhe", "random_fof"} <= FIXED_EDGE_SWAP_METHODS
