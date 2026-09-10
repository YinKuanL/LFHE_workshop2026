import argparse
import csv
import importlib.util
from pathlib import Path

import networkx as nx

import main


ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "static_random", "epidemic", "dissdl", "random_fof", "morph", "lfhe",
    "lfhe_expand",
)


def config(method: str, n: int = 50, seed: int = 42):
    return main.make_config(main.parser().parse_args([
        "--method", method,
        "--num-clients", str(n),
        "--seed", str(seed),
        "--protocol", "scalable",
        "--dmax", "4",
        "--initial-graph", "bounded_connected",
        "--shared-initial-topology",
        "--output-dir", "unused",
    ]))


def undirected_edges(graph):
    return {tuple(sorted((int(left), int(right)))) for left, right in graph.to_undirected().edges()}


def test_cli_parsers_support_python_without_boolean_optional_action(monkeypatch):
    monkeypatch.delattr(argparse, "BooleanOptionalAction", raising=False)
    parsed = main.parser().parse_args([
        "--method", "lfhe",
        "--num-clients", "50",
        "--seed", "42",
        "--protocol", "scalable",
        "--output-dir", "unused",
        "--shared-initial-topology",
    ])
    assert parsed.shared_initial_topology is True
    assert main.batch_parser().parse_args(["--run-all"]).continue_on_error is True
    assert main.batch_parser().parse_args(
        ["--run-all", "--no-continue-on-error"]
    ).continue_on_error is False


def test_every_main_method_has_the_exact_static_random_initial_topology():
    for n in (50, 100, 200, 500):
        for seed in range(42, 47):
            expected = undirected_edges(main.bounded_connected(n, 4, seed))
            hashes = set()
            for method in METHODS:
                graph, pac_state, common = main.initialize_topology(config(method, n, seed))
                assert undirected_edges(common) == expected
                assert undirected_edges(graph) == expected
                assert nx.is_connected(graph.to_undirected())
                hashes.add(main.undirected_topology_hash(common))
                if method in {"random_fof", "lfhe"}:
                    assert pac_state is not None
                    assert pac_state.edge_count == len(expected)
                    assert pac_state.fixed_edge_count == len(expected)
                if method == "lfhe_expand":
                    assert pac_state is not None
                    assert pac_state.edge_count == len(expected)
                    assert pac_state.edge_budget == n * 4 // 2
                    assert pac_state.fixed_edge_count is None
            assert len(hashes) == 1


def test_directed_methods_preserve_the_common_receiver_neighborhood():
    expected = main.bounded_connected(50, 4, 42)
    for method in ("epidemic", "dissdl", "morph"):
        graph, pac_state, _ = main.initialize_topology(config(method))
        assert pac_state is None and graph.is_directed()
        for client in graph:
            assert set(graph.predecessors(client)) == set(expected.neighbors(client))
            assert graph.in_degree(client) <= 4
        assert graph.number_of_edges() == 2 * expected.number_of_edges()


def test_morph_and_epidemic_use_the_common_graph_at_round_zero():
    morph_cfg = config("morph", n=10)
    morph_graph, _, common = main.initialize_topology(morph_cfg)
    nodes = main.make_morph_nodes(main.initial_states(10, 42), morph_graph, morph_cfg)
    assert [node.in_degree for node in nodes] == [
        common.degree(client) for client in range(10)
    ]
    assert all(node.validate_state() is None for node in nodes)

    epidemic_cfg = config("epidemic", n=50)
    epidemic_graph, _, common = main.initialize_topology(epidemic_cfg)
    round_zero = main.epidemic_graph_for_round(epidemic_cfg, epidemic_graph, 0)
    round_one = main.epidemic_graph_for_round(epidemic_cfg, epidemic_graph, 1)
    assert undirected_edges(round_zero) == undirected_edges(common)
    assert set(dict(round_one.out_degree()).values()) == {4}


def test_shared_main_manifest_is_complete_distinct_and_training_matched():
    source = list(csv.DictReader(
        (ROOT / "manifests" / "workshop_main_fresh_md_aligned_n50_500.csv").open()
    ))
    shared = list(csv.DictReader(
        (ROOT / "manifests" / "workshop_main_shared_static_init_n50_500.csv").open()
    ))
    assert len(source) == len(shared) == 120
    assert all(row["shared_initial_topology"] == "true" for row in shared)
    assert all(row["output_dir"].startswith(
        "outputs/workshop_main_shared_static_init_n50_500/"
    ) for row in shared)
    ignored = {"suite", "output_dir", "shared_initial_topology"}
    for old, new in zip(source, shared):
        assert {key: value for key, value in old.items() if key not in ignored} == {
            key: value for key, value in new.items() if key not in ignored
        }


def test_manifest_generator_and_runner_contract():
    spec = importlib.util.spec_from_file_location(
        "shared_manifest_generator",
        ROOT / "scripts" / "generate_shared_initial_main_manifest.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.render(module.generate()) == (
        ROOT / "manifests" / "workshop_main_shared_static_init_n50_500.csv"
    ).read_text(encoding="utf-8-sig")
    assert "--shared-initial-topology" in (ROOT / "run_manifest_row.py").read_text()
